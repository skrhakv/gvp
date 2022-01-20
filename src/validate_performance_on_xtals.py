from models import MQAModel
from util import load_checkpoint
import tensorflow as tf
import mdtraj as md
import numpy as np
import os
from glob import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

abbrev = {"ALA" : "A" , "ARG" : "R" , "ASN" : "N" , "ASP" : "D" , "CYS" : "C" , "CYM" : "C", "GLU" : "E" , "GLN" : "Q" , "GLY" : "G" , "HIS" : "H" , "ILE" : "I" , "LEU" : "L" , "LYS" : "K" , "MET" : "M" , "PHE" : "F" , "PRO" : "P" , "SER" : "S" , "THR" : "T" , "TRP" : "W" , "TYR" : "Y" , "VAL" : "V"}
lookup = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9, 'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8, 'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 'N': 2, 'Y': 18, 'M': 12}

def process_struc(strucs):
    """Takes a list of single frame md.Trajectory objects
    """

    pdbs = []
    for s in strucs:
        prot_iis = s.top.select("protein and (name N or name CA or name C or name O)")
        prot_bb = s.atom_slice(prot_iis)
        pdbs.append(prot_bb)

    B = len(strucs)
    L_max = np.max([pdb.top.n_residues for pdb in pdbs])
    print(L_max)
    X = np.zeros([B, L_max, 4, 3], dtype=np.float32)
    S = np.zeros([B, L_max], dtype=np.int32)

    for i, prot_bb in enumerate(pdbs):
        l = prot_bb.top.n_residues
        xyz = prot_bb.xyz.reshape(l, 4, 3)

        seq = [r.name for r in prot_bb.top.residues]
        S[i, :l] = np.asarray([lookup[abbrev[a]] for a in seq], dtype=np.int32)
        X[i] = np.pad(xyz, [[0,L_max-l], [0,0], [0,0]],
                      'constant', constant_values=(np.nan, ))

    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[isnan] = 0.
    X = np.nan_to_num(X)

    return X, S, mask

def predict(model, nn_path, strucs, opt=tf.keras.optimizers.Adam()):
    load_checkpoint(model, opt, nn_path)
    X, S, mask = process_struc(strucs)
    # model.multiclass = True
    prediction = model(X, S, mask, train=False, res_level=True)
    return prediction

# Load val set
val_set = np.load('/project/bowmanlab/borowsky.jonathan/FAST-cs/new_pockets/labels/new_pocket_labels_validation.npy',
                  allow_pickle=True)
strucs = [md.load(p[0]) for p in val_set]

# Create model
DROPOUT_RATE = 0.1
NUM_LAYERS = 4
model = MQAModel(node_features=(8,50), edge_features=(1,32), hidden_dim=(16,100),
                 num_layers=NUM_LAYERS, dropout=DROPOUT_RATE, multiclass=True)


for fold in range(0, 5):
    nn_dir = ("/project/bowmanlab/ameller/gvp/5-fold-cv-window-40-nearby-pv-procedure/"
              "net_8-50_1-32_16-100_lr_0.0002_dr_0.1_nl_4_hd_100_b1_40epoch_feat_method_"
              f"nearby-pv-procedure_rank_7_stride_1_pos116_neg20_window_40ns_multiclass_fold_{fold}")

    # Determine which network to use (i.e. epoch with best AUC)
    # note code assumes 20 training epochs
    auc = []
    for epoch in range(40):
        auc.append(np.load(f"{nn_dir}/val_auc_{epoch}.npy").item())
    best_epoch = np.argmax(auc)

    # Determine network name
    index_filenames = glob(f"{nn_dir}/*.index")
    nn_id = os.path.basename(index_filenames[0]).split('_')[0]
    nn_path = f"{nn_dir}/{nn_id}_{str(best_epoch).zfill(3)}"

    prediction = predict(model, nn_path, strucs)
    np.save(f'{nn_dir}/xtal_val_set_predictions.npy', prediction)

# This is the path to the trained model. You should have something like
# 1623688914_049.index and 1623688914_049.data-00000-of-00001 in the 'models' dir
# INSERT BEST MODEL HERE
# nn_dir = ("net_8-50_1-32_16-100_nl_4_lr_0.00029_dr_0.13225_b1_"
#           "20epoch_feat_method_nearby-pv-procedure_rank_7_stride_1"
#           "_pos116_neg20_window_40ns_test_TEM-1MY0-1BSQ-nsp5-il6-2OFV")
# nn_path = (f"/project/bowmanlab/ameller/gvp/{nn_dir}/"
#            "1636056869_018")

# nn_path = "/project/bowmore/ameller/gvp/models/1623688914_049"
# opt = tf.keras.optimizers.Adam()

# prediction = predict(model, nn_path, strucs)
# np.save(f'../data/val-set-predictions-{nn_dir}.npy', prediction)
