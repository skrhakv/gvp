#!/bin/bash
#BSUB -L /bin/bash
#BSUB -q bowman
#BSUB -R "select[defined(rocm)] rusage[rocm=1]"
#BSUB -o log/training-%J.log
#BSUB -e log/training-%J.log

# things to do before Singularity
sleep $[ ( $RANDOM % 30 )  + 10 ]s

export HIP_VISIBLE_DEVICES=$(/opt/rocm-common/bin/rocm_smi_lsf.py -n 1)
echo "HIP ROCM REQUESTS $HIP_VISIBLE_DEVICES"

#The Singularity run
singularity exec --no-home -H /project/bowmanlab/mdward/tf-singularity-home/ -B /project:/project /project/bowmanlab/rocm/tensorflow-rocm.sif /bin/bash /project/bowmanlab/mdward/projects/FAST-pocket-pred/gvp/src/train-submit-amd.sh

#Things to do after singularity
echo Done
