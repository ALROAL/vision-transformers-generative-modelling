#!/bin/sh
#BSUB -q gpuv100
#BSUB -J ViTCVAE
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 05:00
#BSUB -R "rusage[mem=8GB]"
##BSUB -R "select[gpu32gb]" #options gpu16gb or gpu32gb
#BSUB -o outputs/gpu_%J.out
#BSUB -e outputs/gpu_%J.err
# -- end of LSF options --

nvidia-smi

source aml/bin/activate

python3 main.py --name ViTCVAE_R --model-type ViTCVAE_R --dim 512 --mlp_dim 512 --batch_size 512 --depth 5 --ngf 32 --max-epochs 100 --num-workers 8 >| outputs/ViTCVAE_R.out 2>| error/ViTCVAE_R.err
