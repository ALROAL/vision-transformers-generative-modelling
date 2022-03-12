#!/bin/sh
#BSUB -q hpc
#BSUB -J test
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -W 00:15
#BSUB -R "rusage[mem=16GB]"
#BSUB -o outputs/gpu_%J.out
#BSUB -e outputs/gpu_%J.err
# -- end of LSF options --

module load python3
source ../envs/aml/bin/activate

# Name - Model - Epochs

python3 main.py Classifier_test1 Classifier 2 >| outputs/class_test.out 2>| error/class_test.err

