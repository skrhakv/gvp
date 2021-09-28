import os

import mlflow
import optuna
import tensorflow as tf
from tensorflow import keras as keras

from datasets import *
from models import *

NUM_EPOCHS = 20
pos_thresh = 20
neg_thresh = 5

## Vary throughout task 1 training ##
BATCH_SIZE = 1
window = 10

# Input data specs
min_rank = 6
test_string = 'TEM-1MY0-1BSQ-nsp5-il6'
featurization_method = 'gp-to-nearest-resi'
stride = 1
FILESTEM = f'{featurization_method}-min-rank-{min_rank}-window-{window}-stride-{stride}-test-{test_string}'
# Output data directory
outdir = "/project/bowmanlab/ameller/gvp/optuna/"

loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

def make_model(dropout_rate, num_layers, hidden_dim):
    model = MQAModel(
        node_features=(8, 50),
        edge_features=(1, 32),
        hidden_dim=hidden_dim,
        dropout=dropout_rate,
        num_layers=num_layers,
    )
    return model


def main_cv(learning_rate, dropout_rate, num_layers, hidden_layers, cv_fold):

    trainset, valset, testset = pockets_dataset(BATCH_SIZE, FILESTEM, cv_fold)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model = make_model(
        dropout_rate=dropout_rate, num_layers=num_layers, hidden_dim=(16, hidden_layers)
    )

    loop_func = loop
    best_epoch, best_val = 0, np.inf
    val_losses = []

    for epoch in range(NUM_EPOCHS):
        loss = loop_func(trainset, model, train=True, optimizer=optimizer)
        print("EPOCH {} training loss: {}".format(epoch, loss))
        loss = loop_func(valset, model, train=False, val=False)
        val_losses.append(loss)
        print(" EPOCH {} validation loss: {}".format(epoch, loss))
        if loss < best_val:
            best_epoch, best_val = epoch, loss
        print("EPOCH {} VAL {:.4f}".format(epoch, loss))

    # Save out validation losses
    np.save(f"{outdir}/{cv_fold}_cv_loss.npy", val_losses)

    return best_val


def loop(dataset, model, train=False, optimizer=None, alpha=1, val=False):

    losses = []
    y_pred, y_true, meta_d, targets = [], [], [], []
    batch_num = 0
    # for batch in tqdm.tqdm(dataset):
    for batch in dataset:
        X, S, y, meta, M = batch
        if train:
            with tf.GradientTape() as tape:
                prediction = model(X, S, M, train=True, res_level=True)
                # Grab balanced set of residues
                iis = choose_balanced_inds(y)
                y = tf.gather_nd(y, indices=iis)
                y = y >= pos_thresh
                y = tf.cast(y, tf.float32)
                prediction = tf.gather_nd(prediction, indices=iis)
                loss_value = loss_fn(y, prediction)
        else:
            # we set train to False
            # test and validation sets (select all examples except for intermediate values)
            prediction = model(X, S, M, train=False, res_level=True)
            iis = convert_test_targs(y)
            y = tf.gather_nd(y, indices=iis)
            y = y >= pos_thresh
            y = tf.cast(y, tf.float32)
            prediction = tf.gather_nd(prediction, indices=iis)
            loss_value = loss_fn(y, prediction)
        if train:
            assert np.isfinite(float(loss_value))
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

        losses.append(float(loss_value))
        y_pred.extend(prediction.numpy().tolist())
        y_true.extend(y.numpy().tolist())

        batch_num += 1

    return np.mean(losses)


def convert_test_targs(y):
    # Need to convert targs (volumes) to 1s and 0s but also discard
    # intermediate values
    iis_pos = [np.where(np.array(i) >= pos_thresh)[0] for i in y]
    iis_neg = [np.where((np.array(i) < neg_thresh) & (np.array(i) >= 0))[0] for i in y]
    iis = []
    count = 0
    for i, j in zip(iis_pos, iis_neg):
        subset_iis = [[count, s] for s in j]
        for pair in subset_iis:
            iis.append(pair)
        subset_iis = [[count, s] for s in i]
        for pair in subset_iis:
            iis.append(pair)
        count += 1

    return iis


def choose_balanced_inds(y):
    iis_pos = [np.where(np.array(i) >= pos_thresh)[0] for i in y]
    iis_neg = [np.where((np.array(i) < neg_thresh) & (np.array(i) > -1))[0] for i in y]
    count = 0
    iis = []
    for i, j in zip(iis_pos, iis_neg):
        # print(len(i),len(j))
        if len(i) < len(j):
            subset = np.random.choice(j, len(i), replace=False)
            subset_iis = [[count, s] for s in subset]
            for pair in subset_iis:
                iis.append(pair)
            subset_iis = [[count, s] for s in i]
            for pair in subset_iis:
                iis.append(pair)
        elif len(j) < len(i):
            subset = np.random.choice(i, len(j), replace=False)
            subset_iis = [[count, s] for s in subset]
            for pair in subset_iis:
                iis.append(pair)
            subset_iis = [[count, s] for s in j]
            for pair in subset_iis:
                iis.append(pair)
        else:
            subset_iis = [[count, s] for s in j]
            for pair in subset_iis:
                iis.append(pair)
            subset_iis = [[count, s] for s in i]
            for pair in subset_iis:
                iis.append(pair)

        count += 1
    # hacky way to deal with situation when there are no positive examples (or negative)
    # for a given structure
    if len(iis) == 0:
        iis = [[0, 0]]

    return iis


def mlflow_callback(study, trial):
    "Saves the parameters for each optuna run, along with the best loss for each"
    trial_value = trial.value if trial.value is not None else float("nan")
    with mlflow.start_run(run_name=f"optuna_CV"):
        mlflow.log_params(trial.params)
        mlflow.log_metrics({"loss": trial_value})


def train_function(train_op, results, sess, cv_fold):
    '''
        Callable object that is run by a thread
        train_op : a callable function passed to tf Session
        results : dictionary storing cv loss for multiple threads
    '''
    cv_loss = sess.run(train_op)
    results[cv_fold] = cv_loss


def objective(trial):
    "Defines objective function for optuna, uses mean best vallidation loss across folds"

    learning_rate = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    dropout_rate = trial.suggest_loguniform("dropout", 1e-4, 1e-2)
    num_layers = trial.suggest_categorical("num_layers", [3, 4, 5, 6])
    hidden_layers = trial.suggest_categorical("hid_layers", [50, 100, 200])

    loss = np.inf

    sess = tf.compat.v1.Session()

    "This loops computes loss from cross validation, alternatively you can use simply a single test splits, this is done in the GVP paper"
    train_ops = []
    for cv_fold in range(0, 5):
        "Here parallelize training by sending to different GPUs."
        with tf.device("/gpu:%d" % cv_fold):
            train_ops.append(
                main_cv(
                    learning_rate=learning_rate,
                    dropout_rate=dropout_rate,
                    num_layers=num_layers,
                    hidden_layers=hidden_layers,
                    cv_fold=cv_fold,)
            )
            # loss = main_cv(
            #     learning_rate=learning_rate,
            #     dropout_rate=dropout_rate,
            #     num_layers=num_layers,
            #     hidden_layers=hidden_layers,
            #     cv_fold=cv_fold,
            # )
            # cv_losses.append(loss)


    # Create multiple training threads

    # need a mutable object to pass to thread constructor
    # that can store results of training run
    cv_losses = {}

    train_threads = []
    for cv_fold, train_op in enumerate(train_ops):
        train_threads.append(threading.Thread(target=train_function,
                                              args=(train_op, results, sess, cv_fold)))

    # Start threads and block on their completion
    for t in train_threads:
        t.start()
    for t in train_threads:
        t.join()

    # Returns mean val loss across CV folds
    return np.mean([loss for loss in cv_losses.values()])


if __name__ == "__main__":
    #### GPU INFO ####
    #tf.debugging.enable_check_numerics()
    os.makedirs(outdir, exist_ok=True)
    model_path = outdir + '{}_{}'

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    study = optuna.create_study()
    study.optimize(objective, n_trials=15, callbacks=[mlflow_callback])

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))