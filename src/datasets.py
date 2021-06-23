import tensorflow as tf
import numpy as np
import mdtraj as md
import pandas as pd
import glob
import os


abbrev = {"ALA" : "A" , "ARG" : "R" , "ASN" : "N" , "ASP" : "D" , "CYS" : "C" , "CYM" : "C", "GLU" : "E" , "GLN" : "Q" , "GLY" : "G" , "HIS" : "H" , "ILE" : "I" , "LEU" : "L" , "LYS" : "K" , "MET" : "M" , "PHE" : "F" , "PRO" : "P" , "SER" : "S" , "THR" : "T" , "TRP" : "W" , "TYR" : "Y" , "VAL" : "V"}
lookup = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9, 'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8, 'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 'N': 2, 'Y': 18, 'M': 12}

DATA_DIR = "/project/bowmanlab/mdward/projects/FAST-pocket-pred/gvp/data"

def pockets_dataset(batch_size):
    #will have [(xtc,pdb,index,residue,1/0),...]
    X_train = np.load(os.path.join(DATA_DIR,"X_train-TEM-VP35.npy"))
    y_train = np.load(os.path.join(DATA_DIR,"y_train-TEM-VP35.npy"),allow_pickle=True)
    trainset = list(zip(X_train,y_train))   

    X_validate = np.load(os.path.join(DATA_DIR,"X_validate-TEM-VP35.npy"))
    y_validate = np.load(os.path.join(DATA_DIR,"y_validate-TEM-VP35.npy"),allow_pickle=True)
    valset = list(zip(X_validate,y_validate))    

    X_test = np.load(os.path.join(DATA_DIR,"X_test-TEM-VP35.npy"))
    y_test = np.load(os.path.join(DATA_DIR,"y_test-TEM-VP35.npy"),allow_pickle=True)
    testset = list(zip(X_test, y_test))    

    trainset = DynamicLoader(trainset, batch_size)
    valset = DynamicLoader(valset, batch_size)
    testset = DynamicLoader(testset, batch_size)
    
    output_types = (tf.float32, tf.int32, tf.int32, tf.string, tf.float32)
    trainset = tf.data.Dataset.from_generator(trainset.__iter__, output_types=output_types).prefetch(3)
    valset = tf.data.Dataset.from_generator(valset.__iter__, output_types=output_types).prefetch(3)
    testset = tf.data.Dataset.from_generator(testset.__iter__, output_types=output_types).prefetch(3)
    
    return trainset, valset, testset

def parse_batch(batch):
    #Batch will have [(xtc,pdb,index,residue,1/0),...]
    pdbs = []
    #can parallelize to improve speed
    for ex in batch:
        x, y = ex
        pdb = md.load(x[1])
        prot_iis = pdb.top.select("protein and (name N or name CA or name C or name O)")
        prot_bb = pdb.atom_slice(prot_iis)
        pdbs.append(prot_bb)
    
    B = len(batch)
    L_max = np.max([pdb.top.n_residues for pdb in pdbs])
    X = np.zeros([B, L_max, 4, 3], dtype=np.float32)
    S = np.zeros([B, L_max], dtype=np.int32)
    # -1 so we can distinguish 0 pocket volume and padded indices later 
    y = np.zeros([B, L_max], dtype=np.int32)-1 

    meta = []
    for i,ex in enumerate(batch):
        x, targs = ex
        traj_fn, pdb_fn, traj_iis = x
        traj_iis = int(traj_iis)
        
        pdb = md.load(pdb_fn)
        struc = md.load_frame(traj_fn,traj_iis,top=pdb)
        prot_iis = struc.top.select("protein and (name N or name CA or name C or name O)")
        prot_bb = struc.atom_slice(prot_iis)
        l = prot_bb.top.n_residues
        xyz = prot_bb.xyz.reshape(l,4,3)
        
        seq = [r.name for r in prot_bb.top.residues]
        S[i, :l] = np.asarray([lookup[abbrev[a]] for a in seq], dtype=np.int32)
        X[i] = np.pad(xyz, [[0,L_max-l], [0,0], [0,0]],
                        'constant', constant_values=(np.nan, ))
        y[i, :l] = targs
        meta.append(x)
    
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[isnan] = 0.
    X = np.nan_to_num(X)
        
    return X, S, y, meta, mask
    
class DynamicLoader(): 
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def chunks(self,arr, chunk_size):
        """Yield successive chunk_size chunks from arr."""
        for i in range(0, len(arr), chunk_size):
            yield arr[i:i + chunk_size]
        
    def batch(self):
        dataset = self.dataset
        np.random.shuffle(dataset)
        self.clusters = list(self.chunks(dataset,self.batch_size))

    def __iter__(self):
        self.batch()
        if self.shuffle: np.random.shuffle(self.clusters)
        N = len(self.clusters)
        for batch in self.clusters[:N]:
            yield parse_batch(batch)


