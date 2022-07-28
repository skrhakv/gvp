import tensorflow as tf
from models import MQAModel
import numpy as np
from glob import glob
import mdtraj as md
import os

from validate_performance_on_xtals import process_strucs, predict_on_xtals

def make_predictions(pdb_paths, model, nn_path, debug=False, output_basename=None):
    '''
        pdb_paths : list of pdb paths
        model : MQAModel corresponding to network in nn_path
        nn_path : path to checkpoint files
    '''
    strucs = [md.load(s) for s in pdb_paths]
    X, S, mask = process_strucs(strucs)
    if debug:
        np.save(f'{output_basename}_X.npy', X)
        np.save(f'{output_basename}_S.npy', S)
        np.save(f'{output_basename}_mask.npy', mask)
    predictions = predict_on_xtals(model, nn_path, X, S, mask)
    return predictions

# main method
if __name__ == '__main__':
    # inputs
    nn_dir = '/project/bowmanlab/ameller/gvp/task2/train-with-4-residue-batches-no-balancing-intermediates-in-training/net_8-50_1-32_16-100_dr_0.1_nl_4_hd_100_lr_2e-05_b4resis_b1proteins_20epoch_feat_method_gp-to-nearest-resi-procedure_rank_7_stride_1_window_40_pos_20_refine_feat_method_fpocket_drug_scores_max_window_40_cutoff_0.3_stride_5'
    best_epoch = 1

    # TO DO - provide pdbs
    strucs = [
        '../data/ACE2.pdb',
    ]
    output_name = 'ACE2'
    output_folder = '/project/bowmore/ameller/projects/pocket_prediction/data'

    # debugging mode can be turned on to output protein features and sequence
    debug = False

    # Determine path to checkpoint files
    index_filenames = glob(f"{nn_dir}/*.index")
    nn_id = os.path.basename(index_filenames[0]).split('_')[0]
    nn_path = f"{nn_dir}/{nn_id}_{str(best_epoch).zfill(3)}"

    nn_path = "../models/pocketminer"
    print(f'using network {nn_path}')

    # MQA Model used for selected NN network
    DROPOUT_RATE = 0.1
    NUM_LAYERS = 4
    HIDDEN_DIM = 100
    model = MQAModel(node_features=(8, 50), edge_features=(1, 32),
                     hidden_dim=(16, HIDDEN_DIM),
                     num_layers=NUM_LAYERS, dropout=DROPOUT_RATE)
    
    
    if debug:
        output_basename = f'{output_folder}/{output_name}'
        predictions = make_predictions(strucs, model, nn_path, debug=True, output_basename=output_basename)
    else:
        predictions = make_predictions(strucs, model, nn_path)

    # output filename can be modified here
    np.save(f'{output_folder}/{output_name}-preds.npy', predictions)
    np.savetxt(os.path.join(output_folder,f'{output_name}-predictions.txt'), predictions, fmt='%.4g', delimiter='\n')


