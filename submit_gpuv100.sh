#!/bin/sh
#BSUB -q gpuv100
#BSUB -J ViTVAE
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 01:00
#BSUB -R "rusage[mem=8GB]"
##BSUB -R "select[gpu32gb]"
#BSUB -o outputs/gpu_%J.out
#BSUB -e outputs/gpu_%J.err
# -- end of LSF options --

# nvidia-smi
# Load the cuda module
# module load cuda/10.2

# /appl/cuda/10.2/samples/NVIDIA_CUDA-10.2_Samples/bin/x86_64/linux/release/deviceQuery

source aml/bin/activate

# Options
# --name : Name for wandb
# --model-type : Model type to train either ViTVAE or Classifier
# --max-epochs : Number of max epochs
# --num-workers : Number of threads use in loading data (should almost always be the same as BSUB -n)
# --dim : Last dimension of output tensor after linear transformation
# --depth : Number of Transformer blocks
# --heads : Number of heads in Multi-head Attention layer
# --mlp_dim : Dimension of the MLP (FeedForward) layer

# python3 main.py --name Classifier --model-type Classifier --max-epochs 200 --num-workers 8 >| outputs/class.out 2>| error/class.err

python3 main.py --name ViTVAE --model-type ViTVAE --max-epochs 200 --num-workers 8 >| outputs/ViTVAE.out 2>| error/ViTVAE.err

