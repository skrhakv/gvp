#!/bin/bash
#BSUB -L /bin/bash
#BSUB -q bowman
#BSUB -R "select[defined(rocm)] rusage[rocm=1]"
#BSUB -J training
#BSUB -o logs/training-%J.log
#BSUB -e logs/training-%J.log

# things to do before Singularity
# sleep 7m
export HIP_VISIBLE_DEVICES=$(/opt/rocm-common/bin/find_rocm.sh -n 1)
date
hostname
echo "HIP ROCM REQUESTS $HIP_VISIBLE_DEVICES"

#The Singularity run
singularity exec --no-home -H /project/bowmanlab/ameller/tf-singularity-home/ -B /project:/project /project/bowmanlab/rocm/tensorflow-rocm.sif python train_fold_pockets_residue_batches.py training-yaml-files/to-run/train_4prot_4resi_no_balance_intermediates_True_fold_4.yaml

#Things to do after singularity
echo Done
