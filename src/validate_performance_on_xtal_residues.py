from models import MQAModel
from util import load_checkpoint
import tensorflow as tf
import mdtraj as md
import numpy as np
import os
from glob import glob
from tensorflow import keras as keras
from tqdm import tqdm
from validate_performance_on_xtals import process_strucs, process_paths, predict_on_xtals


if __name__ == '__main__':
    label_dictionary = np.load('/project/bowmore/ameller/projects/pocket_prediction/data/val_label_dictionary.npy',
                               allow_pickle=True).item()
    val_set_apo_ids = np.load('/project/bowmore/ameller/projects/pocket_prediction/data/val_apo_ids.npy')
    val_set_apo_ids_with_chainids = np.load('/project/bowmore/ameller/projects/'
                                            'pocket_prediction/data/val_apo_ids_with_chainids.npy')

    l_max = max([len(l) for l in label_dictionary.values()])

    label_mask = np.zeros([len(label_dictionary.keys()), l_max], dtype=bool)
    for i, l in enumerate(label_dictionary.values()):
        label_mask[i] = np.pad(l != 2, [[0, l_max - len(l)]])

    true_labels = np.zeros([len(label_dictionary.keys()), l_max], dtype=int)
    for i, l in enumerate(label_dictionary.values()):
        true_labels[i] = np.pad(l, [[0, l_max - len(l)]])

    print(np.sum(true_labels[label_mask] == 0))
    print(np.sum(true_labels[label_mask] == 1))

    # Create model
    DROPOUT_RATE = 0.1
    NUM_LAYERS = 4
    HIDDEN_DIM = 100

    # get NN directories
    nn_dirs = glob('/project/bowmanlab/ameller/gvp/task2/*/*')

    X, S, mask = process_paths(val_set_apo_ids_with_chainids, use_tensors=False)

    for nn_dir in nn_dirs:
        # if sidechain is in name, need to use tensors
        if 'sidechain' in nn_dir:
            continue

        # determine number of compeleted epochs
        val_files = glob(f"{nn_dir}/val_pr_auc_*.npy")

        if len(val_files) == 0:
            continue

        # Determine network name
        index_filenames = glob(f"{nn_dir}/*.index")
        nn_id = os.path.basename(index_filenames[0]).split('_')[0]

        model = MQAModel(node_features=(8, 50), edge_features=(1, 32),
                         hidden_dim=(16, HIDDEN_DIM),
                         num_layers=NUM_LAYERS, dropout=DROPOUT_RATE)

        # Determine which network to use (i.e. epoch with best AUC)
        pr_auc = []
        auc = []

        auc_metric = keras.metrics.AUC(name='auc')
        pr_auc_metric = keras.metrics.AUC(curve='PR', name='pr_auc')

        for epoch in tqdm(range(len(val_files))):
            nn_path = f"{nn_dir}/{nn_id}_{str(epoch).zfill(3)}"
            predictions = predict_on_xtals(model, nn_path, X, S, mask)

            y_pred = predictions[mask.astype(bool) & label_mask]
            y_true = true_labels[label_mask]

            auc_metric.update_state(y_true, y_pred)
            pr_auc_metric.update_state(y_true, y_pred)

            np.save(os.path.join(nn_dir, f"val_new_auc_{epoch}.npy"), auc_metric.result().numpy())
            np.save(os.path.join(nn_dir, f"val_new_pr_auc_{epoch}.npy"), pr_auc_metric.result().numpy())
            np.save(os.path.join(nn_dir, f"val_new_y_pred_{epoch}.npy"), y_pred)
            np.save(os.path.join(nn_dir, f"val_new_y_true_{epoch}.npy"), y_true)

            pr_auc.append(pr_auc_metric.result().numpy())
            auc.append(auc_metric.result().numpy())

        print(auc)
        best_epoch = np.argmax(auc)
        nn_path = f"{nn_dir}/{nn_id}_{str(best_epoch).zfill(3)}"

        predictions = predict_on_xtals(model, nn_path, X, S, mask)

        np.save(f'{nn_dir}/new_val_set_y_pred_best_epoch.npy', predictions)

