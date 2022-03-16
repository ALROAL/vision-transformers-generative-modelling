#!/bin/sh
#BSUB -q gpua100
#BSUB -J Classifier
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 01:00
#BSUB -R "rusage[mem=8GB]"
##BSUB -R "select[gpu40gb]" #options gpu40gb or gpu80gb
#BSUB -o outputs/gpu_%J.out
#BSUB -e outputs/gpu_%J.err
# -- end of LSF options --

nvidia-smi

source aml/bin/activate

# Options
# --name : Name for wandb
# --model-type : Model type to train either ViTVAE, Classifier, or Classifier_deep
# --max-epochs : Number of max epochs
# --num-workers : Number of threads use in loading data (should almost always be the same as BSUB -n)
# --dim : Last dimension of output tensor after linear transformation
# --depth : Number of Transformer blocks
# --heads : Number of heads in Multi-head Attention layer
# --mlp_dim : Dimension of the MLP (FeedForward) layer
# --lr : Learning rate (Currently only for classifier)
# --patch-size : Number of patches. (patch_size can't be more than 8 for cifar10) image_size must be divisible by patch_size. The number of patches is: n = (image_size // patch_size) ** 2 and n must be greater than 16. 

python3 main.py --name ViT-Large --model-type Classifier_deep --max-epochs 300 --num-workers 8 --depth 16 --lr 1e-4 --dim 1024 --mlp_dim 2048 --heads 16 --patch_size 8 >| outputs/class.out 2>| error/class.err

# python3 main.py --name ViTVAE --model-type ViTVAE --max-epochs 200 --num-workers 8 >| outputs/ViTVAE.out 2>| error/ViTVAE.err
