#!/bin/sh
#BSUB -q gpuv100
#BSUB -J ViTVAE
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 04:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "select[gpu16gb]"
#BSUB -o outputs/gpu_%J.out
#BSUB -e outputs/gpu_%J.err
# -- end of LSF options --

# nvidia-smi
# Load the cuda module
# module load cuda/10.2

# /appl/cuda/10.2/samples/NVIDIA_CUDA-10.2_Samples/bin/x86_64/linux/release/deviceQuery

source aml/bin/activate

# Name - Model - Epochs

# Name : wandb name
# Model : Classifier or ViTVAE
# Epochs : number of max epochs

# python3 main.py Classifier_200 Classifier 200 >| outputs/class.out 2>| error/class.err

python3 main.py ViTVAE_test ViTVAE 200 >| outputs/ViTVAE.out 2>| error/ViTVAE.err

