#!/bin/sh
#BSUB -q gpua100
#BSUB -J Classifier
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 20:00
#BSUB -R "rusage[mem=8GB]"
##BSUB -R "select[gpu40gb]" #options gpu40gb or gpu80gb
#BSUB -o outputs/gpu_%J.out
#BSUB -e outputs/gpu_%J.err
# -- end of LSF options --

nvidia-smi

source aml/bin/activate

# Options
# Run main.py --help to get options

#python3 main.py --name ViT-Base --model-type ViT --optim SGD --max-epochs 200 --num-workers 8 --depth 12 --lr 1e-4 --dim 768 --mlp_dim 3072 --heads 12 --patch-size 8 >| outputs/class.out 2>| error/class.err
#python3 main.py --name ViT-Large --model-type ViT --optim SGD --max-epochs 200 --num-workers 8 --depth 24 --lr 1e-4 --dim 1024 --mlp_dim 4096 --heads 16 --patch-size 8 >| outputs/class.out 2>| error/class.err
#python3 main.py --name ViT-Huge --model-type ViT --optim SGD --max-epochs 200 --num-workers 8 --depth 32 --lr 1e-4 --dim 1280 --mlp_dim 5120 --heads 16 --patch-size 8 >| outputs/class.out 2>| error/class.err


#python3 main.py --name ViTVAE --model-type ViTVAE --max-epochs 100 --num-workers 8 >| outputs/ViTVAE.out 2>| error/ViTVAE.err

#python3 main.py --name ConvCVAE --model-type ConvCVAE --dim 256 --batch_size 64 --max-epochs 100 --num-workers 8 >| outputs/ViTVAE.out 2>| error/ViTVAE.err

#python3 main.py --name ViTCVAE_R --model-type ViTCVAE_R --dim 256 --mlp_dim 256 --batch_size 64 --ngf 32 --max-epochs 100 --num-workers 8 >| outputs/ViTCVAE_R.out 2>| error/ViTCVAE_R.err

python3 main.py --name ViTVAE_PatchGAN_prep_f0_batch512 --model-type ViTVAE_PatchGAN_prepared --batch_size 512 --max-epochs 200 --num-workers 8 >| outputs/ViTVAE_PatchGAN_f0_batch512.out 2>| error/ViTVAE_PatchGAN_f0_batch512.err

#python3 main.py --name ViTVAE_GAN_f03_batch512 --model-type ViTVAE_GAN --batch_size 512 --max-epochs 200 --num-workers 8 >| outputs/ViTVAE_GAN_f03_batch512.out 2>| error/ViTVAE_GAN_f03_batch512.err
