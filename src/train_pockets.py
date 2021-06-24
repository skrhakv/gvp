import tensorflow as tf
#tf.debugging.enable_check_numerics()
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

from datetime import datetime
from datasets import * 
import tqdm, sys
import util, pdb
from tensorflow import keras as keras
from models import *
import os
from util import save_checkpoint, load_checkpoint

models_dir = '../models/{}_{}'

def make_model():
    #Emailed the lead author for what these values should be, these are good defaults.
    model = MQAModel(node_features=(8,50), edge_features=(1,32), hidden_dim=(16,100))
    return model

def main():
    trainset, valset, testset = pockets_dataset(2)# batch size = N proteins
    optimizer = tf.keras.optimizers.Adam()
    model = make_model()
  
    model_id = int(datetime.timestamp(datetime.now()))

    NUM_EPOCHS = 1
    loop_func = loop
    best_epoch, best_val = 0, np.inf
    
    for epoch in range(NUM_EPOCHS):   
        loss = loop_func(trainset, model, train=True, optimizer=optimizer)
        print('EPOCH {} training loss: {}'.format(epoch,loss))
        save_checkpoint(model, optimizer, model_id, epoch)
        print('EPOCH {} TRAIN {:.4f}'.format(epoch, loss))
        #util.save_confusion(confusion)
        loss = loop_func(valset, model, train=False,val=False)
        print(' EPOCH {} validation loss: {}'.format(epoch,loss))
        if loss < best_val:
            #Could play with this parameter here. Instead of saving best NN based on loss
            #we could save it based on precision/auc/recall/etc. 
            best_epoch, best_val = epoch, loss
        print('EPOCH {} VAL {:.4f}'.format(epoch, loss))
        #util.save_confusion(confusion)

  # Test with best validation loss
    np.save("../models/mw_before_load.npy",model.weights)
    path = models_dir.format(str(model_id).zfill(3), str(epoch).zfill(3))
    load_checkpoint(model, optimizer, path)  
    np.save("../models/mw_after_load.npy",model.weights)
    loss, tp, fp, tn, fn, acc, prec, recall, auc, y_pred, y_true, meta_d = loop_func(testset, model, train=False, val=True)
    print('EPOCH TEST {:.4f} {:.4f}'.format(loss, acc))
    #util.save_confusion(confusion)
    return loss, tp, fp, tn, fn, acc, prec, recall, auc, y_pred, y_true, meta_d
    
    
tp_metric = keras.metrics.TruePositives(name='tp')
fp_metric = keras.metrics.FalsePositives(name='fp')
tn_metric = keras.metrics.TrueNegatives(name='tn')
fn_metric = keras.metrics.FalseNegatives(name='fn')
acc_metric = keras.metrics.BinaryAccuracy(name='accuracy')
prec_metric = keras.metrics.Precision(name='precision')
recall_metric = keras.metrics.Recall(name='recall')
auc_metric = keras.metrics.AUC(name='auc')
    
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    
def loop(dataset, model, train=False, optimizer=None, alpha=1,val=False):
    if val:
        tp_metric.reset_states()
        fp_metric.reset_states()
        tn_metric.reset_states()
        fn_metric.reset_states()
        acc_metric.reset_states()
        prec_metric.reset_states()
        recall_metric.reset_states()
        auc_metric.reset_states()
    
    losses = []
    y_pred, y_true, meta_d, targets = [], [], [], []
    batch_num = 0
    for batch in tqdm.tqdm(dataset):
        X, S, y, meta, M = batch
        if train:
            with tf.GradientTape() as tape:
                prediction = model(X, S, M, train=True, res_level=True)
                #Grab balanced set of residues
                iis = choose_balanced_inds(y,40,10)
                print(iis)
                y = tf.gather_nd(y,indices=iis)
                y = y >= 40
                y = tf.cast(y,tf.float32)
                prediction = tf.gather_nd(prediction,indices=iis)
                loss_value = loss_fn(y, prediction)
        else:
            if val:
                prediction = model(X, S, M, train=False, res_level=True)
                iis = convert_test_targs(y,40,10)
                y = tf.gather_nd(y,indices=iis)
                y = y >= 40
                y = tf.cast(y,tf.float32)
                prediction = tf.gather_nd(prediction,indices=iis)
                loss_value = loss_fn(y, prediction) 
                #to be able to identify each y value with its protein and resid
                meta_pairs = [(meta[ind[0]], ind[1]) for ind in iis]
                meta_d.extend(meta_pairs)
            else:
                prediction = model(X, S, M, train=False, res_level=True)
                iis = choose_balanced_inds(y,40,10)
                y = tf.gather_nd(y,indices=iis)
                y = y >= 40
                y = tf.cast(y,tf.float32)
                prediction = tf.gather_nd(prediction,indices=iis)
                loss_value = loss_fn(y, prediction)
        if train:
            assert(np.isfinite(float(loss_value)))
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
        
        losses.append(float(loss_value))
        if batch_num % 5 == 0:
            print(loss_value)
        y_pred.extend(prediction.numpy().tolist())
        y_true.extend(y.numpy().tolist())
        
        if val:
            y = tf.squeeze(y)
            prediction = tf.squeeze(prediction)
            tp_metric.update_state(y,prediction)
            fp_metric.update_state(y,prediction)
            tn_metric.update_state(y,prediction)
            fn_metric.update_state(y,prediction)
            acc_metric.update_state(y,prediction)
            prec_metric.update_state(y,prediction)
            recall_metric.update_state(y,prediction)
            auc_metric.update_state(y,prediction)

        batch_num += 1
    if val:
        tp = tp_metric.result().numpy()
        fp = fp_metric.result().numpy()
        tn = tn_metric.result().numpy()
        fn = fn_metric.result().numpy()
        acc = acc_metric.result().numpy()
        prec = prec_metric.result().numpy()
        recall = recall_metric.result().numpy()
        auc = auc_metric.result().numpy()
    
    if val:
        return np.mean(losses), tp, fp, tn, fn, acc, prec, recall, auc, y_pred, y_true, meta_d
    else:
        return np.mean(losses)

def convert_test_targs(y,pos_thresh,neg_thresh):
#Need to convert targs (volumes) to 1s and 0s but also discard
#intermediate values
    iis_pos = [np.where(np.array(i)>=pos_thresh)[0] for i in y]
    iis_neg = [np.where((np.array(i)<neg_thresh) & (np.array(i)>-1))[0] for i in y]
    iis = []
    count = 0
    for i,j in zip(iis_pos,iis_neg):
        subset_iis = [[count,s] for s in j]
        for pair in subset_iis:
            iis.append(pair)
        subset_iis = [[count,s] for s in i]
        for pair in subset_iis:
            iis.append(pair)
        count+=1

    return iis

def choose_balanced_inds(y,pos_thresh,neg_thresh):
    iis_pos = [np.where(np.array(i)>=pos_thresh)[0] for i in y]
    iis_neg = [np.where((np.array(i)<neg_thresh) & (np.array(i) > -1))[0] for i in y]
    count = 0
    iis = []
    for i,j in zip(iis_pos,iis_neg):
        print(len(i),len(j))
        if len(i) < len(j):
            subset = np.random.choice(j,len(i),replace=False) 
            subset_iis = [[count,s] for s in subset]
            for pair in subset_iis:
                iis.append(pair)
            subset_iis = [[count,s] for s in i]
            for pair in subset_iis:
                iis.append(pair)
        elif len(j) < len(i):
            subset = np.random.choice(i,len(j),replace=False)
            subset_iis = [[count,s] for s in subset]
            for pair in subset_iis:
                iis.append(pair)
            subset_iis = [[count,s] for s in j]
            for pair in subset_iis:
                iis.append(pair)
        else:
            subset_iis = [[count,s] for s in j]
            for pair in subset_iis:
                iis.append(pair)
            subset_iis = [[count,s] for s in i]
            for pair in subset_iis:
                iis.append(pair)
            
        count+=1
    #hacky way to deal with situation when there are no positive examples (or negative)
    #for a given structure
    if len(iis)==0:
        iis =[[0,0]]

    return iis

loss, tp, fp, tn, fn, acc, prec, recall, auc, y_pred, y_true, meta_d = main()
outdir = "./metrics/net_8-50_1-32_16-100_1epoch_b2prot_TEM-VP35-wMeta-saveWeights/"
print(outdir)
os.mkdir(outdir)
np.save(os.path.join(outdir,"loss.npy"),loss)
np.save(os.path.join(outdir,"tp.npy"),tp)
np.save(os.path.join(outdir,"fp.npy"),fp)
np.save(os.path.join(outdir,"tn.npy"),tn)
np.save(os.path.join(outdir,"fn.npy"),fn)
np.save(os.path.join(outdir,"acc.npy"),acc)
np.save(os.path.join(outdir,"prec.npy"),prec)
np.save(os.path.join(outdir,"recall.npy"),recall)
np.save(os.path.join(outdir,"auc.npy"),auc)
np.save(os.path.join(outdir,"y_pred.npy"),y_pred)
np.save(os.path.join(outdir,"y_true.npy"),y_true)
np.save(os.path.join(outdir,"meta_d.npy"),meta_d)
