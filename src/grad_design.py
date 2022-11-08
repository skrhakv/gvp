from models import MQAModel
from validate_performance_on_xtals import process_strucs
from util import load_checkpoint
import datasets
import mdtraj as md
import tensorflow as tf
import numpy as np
from gvp import *

# load and featurize protein
pdb_paths = [
    '../data/ACE2.pdb',
]
strucs = [md.load(s) for s in pdb_paths]

X, S, mask = process_strucs(strucs)

# convert to tensors
X = tf.convert_to_tensor(X)
S = tf.convert_to_tensor(S)

# print(S)

# Load MQA Model used for selected NN network
nn_path = "../models/pocketminer"
DROPOUT_RATE = 0.1
NUM_LAYERS = 4
HIDDEN_DIM = 100
model = MQAModel(node_features=(8, 50), edge_features=(1, 32),
                 hidden_dim=(16, HIDDEN_DIM),
                 num_layers=NUM_LAYERS, dropout=DROPOUT_RATE)

load_checkpoint(model, tf.keras.optimizers.Adam(), nn_path)

# cryptic pocket mask
# would need to adapt this code if we wanted to do this in batches
cryptic_mask = np.zeros((1, S.shape[1]), dtype=bool)
# set residues that are in the relevant cryptic site to True
cryptic_mask[:, 100:200] = True

#  grad = Lambda(lambda x: tf.gradients(x[0],x[1])[0])([loss, I])
with tf.GradientTape() as tape:
    V, E, E_idx = model.features(X, mask)    
    h_S = model.W_s(S)
    V = vs_concat(V, h_S, model.nv, 0)
    h_V = model.W_v(V)
    h_E = model.W_e(E)
    h_V = model.encoder(h_V, h_E, E_idx, mask, train=False)
    h_V_out = model.W_V_out(h_V)
    out = tf.squeeze(model.dense(h_V_out, training=False), -1)
    out = tf.boolean_mask(out, cryptic_mask)

# first assert that inverse of embedding reproduces the input sequence
S_i = tf.linalg.matmul(h_S, tf.linalg.pinv(model.W_s.weights[0]))
embedding_inverse = tf.math.argmax(tf.nn.softmax(S_i), axis=2)
assert tf.math.reduce_all(tf.math.equal(tf.cast(S, dtype=tf.int64), embedding_inverse))

grads = tape.gradient(out, h_S)

# make an update to sequence
update_step_size = 1
new_h_S = h_S + grads * update_step_size
new_h_S_i = tf.linalg.matmul(new_h_S, tf.linalg.pinv(model.W_s.weights[0]))
new_S = tf.math.argmax(tf.nn.softmax(new_h_S_i), axis=2)

# print positions that are changed
reverse_lookup = {
    v: k
    for k, v in datasets.lookup.items()
}

old_modified_positions = tf.gather_nd(S, indices=tf.where(~tf.math.equal(tf.cast(S, dtype=tf.int64), new_S)))
new_modified_positions = tf.gather_nd(new_S, indices=tf.where(~tf.math.equal(tf.cast(S, dtype=tf.int64), new_S)))
for s_o, s_n in zip(old_modified_positions, new_modified_positions):
    print(reverse_lookup[s_o.numpy()], '->', reverse_lookup[s_n.numpy()])

# make prediction using new sequence
new_prediction = model(X, new_S, mask, train=False, res_level=True)
cryptic_pocket_prediction = tf.boolean_mask(new_prediction, cryptic_mask)

# verify that new prediction is larger than the previous one
print(tf.reduce_mean(cryptic_pocket_prediction), tf.reduce_mean(out))
