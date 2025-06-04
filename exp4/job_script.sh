#!/bin/sh
# BSUB -q gpuv100                                  # specify queue
#BSUB -J Experiment_4                 # Job name
#BSUB -o experiment_4%J.out               # output name
#BSUB -e experiment_4%J.err		   # output name (error)

# BSUB -gpu "num=1:mode=exclusive_process"

#BSUB -W 03:00                  # set walltime limit hh:mm
#BSUB -R "rusage[mem=32GB]"     # memory request
#BSUB -R "select[gpu32gb]"      # memory request

#BSUB -n 5                       # number of cores
#BSUB -u s204138@dtu.dk          # email
# #BSUB -B			 # notify when start
#BSUB -N                       # notify when end

cd Masters-thesis
source venv/bin/activate
python3 exp4/unlearn_and_eval_all.py
