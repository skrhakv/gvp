#!/bin/bash
#BSUB -L /bin/bash
#BSUB -q bowman
#BSUB -R "select[defined(rocm)] rusage[rocm=1]"
#BSUB -o log/training-%J.log
#BSUB -e log/training-%J.log

# things to do before Singularity
sleep $[ ( $RANDOM % 10 )  + 2 ]s
export HIP_PCI_DEVICES=$(/opt/rocm-common/bin/rocm_smi_lsf_pci.py -n 1)
export HIP_VISIBLE_DEVICES=$(/opt/rocm-common/bin/find_rocm.sh -n 1)
echo HIP_VISIBLE_DEVICES is $HIP_VISIBLE_DEVICES from PCI_DEVICES $HIP_PCI_DEVICES
hostname
/opt/rocm-common/bin/hipInfo | egrep 'evice|Bus'
rocm-smi --showpidgpus

#The Singularity run
singularity exec --no-home -H /project/bowmanlab/mdward/tf-singularity-home/ -B /project:/project /project/bowmanlab/rocm/tensorflow-rocm.sif /bin/bash /project/bowmanlab/mdward/projects/FAST-pocket-pred/gvp/src/train-submit-amd.sh

#Things to do after singularity
echo Done
