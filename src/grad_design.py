from models import MQAModel
from validate_performance_on_xtals import process_strucs
from util import load_checkpoint
import mdtraj as md
import tensorflow as tf

# load and featurize protein
pdb_paths = [
    '../data/ACE2.pdb',
]
strucs = [md.load(s) for s in pdb_paths]

X, S, mask = process_strucs(strucs)

# convert to tensors
X = tf.convert_to_tensor(X)
S = tf.convert_to_tensor(S)

print(S)

# Load MQA Model used for selected NN network
nn_path = "../models/pocketminer"
DROPOUT_RATE = 0.1
NUM_LAYERS = 4
HIDDEN_DIM = 100
model = MQAModel(node_features=(8, 50), edge_features=(1, 32),
                 hidden_dim=(16, HIDDEN_DIM),
                 num_layers=NUM_LAYERS, dropout=DROPOUT_RATE)

load_checkpoint(model, tf.keras.optimizers.Adam(), nn_path)

#  grad = Lambda(lambda x: tf.gradients(x[0],x[1])[0])([loss, I])
with tf.GradientTape() as tape:
    prediction = model(X, S, mask, train=False, res_level=True)

grads = tape.gradient(prediction, S)
# grads = tape.gradient(prediction, S)
print(grads)

# make an update to sequence
# S += grads * lr
