import tensorflow as tf
from models import MQAModel

from validate_performance_on_xtals import process_strucs, predict_on_xtals

def make_predictions(strucs, model, nn_path):
'''
	strucs : list of single frame MDTraj trajectories
	model : MQAModel corresponding to network in nn_path
	nn_path : path to checkpoint files
'''
	X, S, mask = process_strucs(strucs)
	predictions = predict_on_xtals(model, nn_path, X, S, mask)
	return predictions

# main method
if __name__ == '__main__':
	# inputs
	nn_dirs = '/project/bowmanlab/ameller/gvp/task2/train-with-4-residue-batches-no-balancing-intermediates-in-training/net_8-50_1-32_16-100_dr_0.1_nl_4_hd_100_lr_2e-05_b4resis_b1proteins_20epoch_feat_method_gp-to-nearest-resi-procedure_rank_7_stride_1_window_40_pos_20_refine_feat_method_fpocket_drug_scores_max_window_40_cutoff_0.3_stride_5'
	best_epoch = 1

	# TO DO - provide pdbs
	strucs = []

	# Determine path to checkpoint files
    index_filenames = glob(f"{nn_dir}/*.index")
    nn_id = os.path.basename(index_filenames[0]).split('_')[0]
    nn_path = f"{nn_dir}/{nn_id}_{str(best_epoch).zfill(3)}"
    print(f'using network {nn_path}')

    # MQA Model used for selected NN network
    model = MQAModel(node_features=(8, 50), edge_features=(1, 32),
                     hidden_dim=(16, HIDDEN_DIM),
                     num_layers=NUM_LAYERS, dropout=DROPOUT_RATE)

	predictions = make_predictions(strucs, model, nn_path)


    # TO DO - decide how to save predictions




