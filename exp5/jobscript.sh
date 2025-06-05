#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J scrub_tsne_box
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 12:00
# request X GB of system-memory
#BSUB -R "rusage[mem=16GB]"

### -- set the email address --
#BSUB s204125@dtu.dk

##BSUB -u s204125@dtu.dk
### -- send notification at start --
#BSUB -B s204125@dtu.dk
### -- send notification at completion--
#BSUB -N s204125@dtu.dk
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o SCRUB_TSNE_BOX_%J.out
#BSUB -e SCRUB_TSNE_BOX_%J.err
# -- end of LSF options --

nvidia-smi
# Load the cuda module
module load python3/3.11.7
module load cuda/11.6

# Activate the virtual environment
source ${VENV_PATH}
# Change to the working directory
cd /work3/s204125/Masters-thesis
source activate.sh 
cd exp5/
python experiment_run_scrub.py