# PocketMiner

PocketMiner is a tool for predicting the locations of cryptic pockets from single protein structures.

PocketMiner is built using the architecture first published in [Learning from Protein Structure with Geometric Vector Perceptrons](https://arxiv.org/abs/2009.01411) by B Jing, S Eismann, P Suriana, RJL Townshend, and RO Dror.

This repository is meant to serve two purposes. If you would like to make predictions on large numbers of structures, we provide a utility script for making predictions. If you would like to use our network for other proteins structure prediction tasks (i.e., use a transfer learning approach), we provide the weights of our final model and instructions on how to fine tune the model for your prediction task.

## Requirements
* UNIX environment
* python==3.7.6
* numpy==1.18.1
* scipy==1.4.1
* pandas==1.0.3
* tensorflow==2.1.0
* tqdm==4.42.1
* mdtraj==1.9.7

We have tested the code with tensorflow versions 2.6.2 and 2.9.1 as well and find that is compatible with these more recent version of tensorflow.

No special hardware is required to run this software.

## Installation
First, clone the PocketMiner repository
```
git clone https://github.com/Mickdub/gvp.git
git switch pocket_pred
```
Then prepare a conda environment (you can use pip and the `linux-requirements.txt` file if you would like) that contains tensorflow, mdtraj, and numpy.
```
conda env create -f pocketminer.yml
conda activate pocketminer
```
There is also a Linux specific `.yml` file called `tf-linux.yml` that may be used by Linux users. Typcially, preparing a conda environment with the required dependencies requires only a minute. Please note that we have removed version number requirements from the `pocketminer.yml` file as we found that this worked better across operating systems. We have tested setup on Linux and MacOS. We will try to validate these instructions for Windows computers shortly.

## Cryptic pocket predictions demo
To use the PocketMiner model, we recommend considering using its web interface (https://pocket-miner-ui.azurewebsites.net/). If however you would like to test the code directly or run predictions on a large number of structures, we suggest modifying the input of `xtal_predict.py` in the `src` directory. There are flags containing `TO DO` that will guide where the code needs to be modified.

There is a demo PDB file in the repository that is used in `xtal_predict.py` (a PDB file of the human protein ACE2). In the code below, simply specify where you would like to send the output. This code will generate a numpy file as well as text file with a prediction for each residue in the input structure:

```
    # TO DO - provide input pdb(s), output name, and output folder
    strucs = [
        '../data/ACE2.pdb',
    ]
    output_name = 'ACE2'
    output_folder = '.'
```
Once you have cloned the repository and created a conda environment, enter the directory and run 
```
cd src
python xtal_predict.py
```
Running this script should only take a few seconds on a standard computer (a GPU is not required for making predictions but is preferred for new model training).

Currently, modifying the `strucs` list to contain multiple structures is not supported (though we will build this functionality in an upcoming update). We recommend looping over the entire main method if you would like to make predictions for multiple structures.


## Transfer learning with the PocketMiner
The following code is needed to create a model object and load the weights from a checkpoint file:

```
from models import MQAModel
from util import load_checkpoint

# MQA Model used for selected NN network
DROPOUT_RATE = 0.1
NUM_LAYERS = 4
HIDDEN_DIM = 100
model = MQAModel(node_features=(8, 50), edge_features=(1, 32),
                 hidden_dim=(16, HIDDEN_DIM),
                 num_layers=NUM_LAYERS, dropout=DROPOUT_RATE)

# Load weights from checkpoint file
nn_path = "models/pocketminer"

load_checkpoint(model, tf.keras.optimizers.Adam(), nn_path)

```

Then, you can continue training the model. Please refer to `train_xtal_predictor.py` for an example of how to write a training loop. There are also multiple class balancing schemes implemented in that file.
