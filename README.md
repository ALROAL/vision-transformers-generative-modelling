Vision Transformers for Generative Modelling
==============================

![test results](https://i.imgur.com/grlnFxo.png)

The aim of this project is to explore the potential of Vision Transformers within the field of generative modelling by comparing the image reconstruction results and performance as generators of CNN-based Conditional Variational Auto-Encoder (CVAE) and ViTbased CVAE models combined with a GAN training for the data augmentation process in an image classification task on CelebFaces Attributes Dataset (CelebA).

- [Quick start](#quick-start)
- [Description](#description)
- [Usage](#usage)
- [Weights & Biases](#weights--biases)

## Quick start

1. Install dependencies
```bash
pip install -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
2. Download data directly from Kaggle [CelebFaces Attributes (CelebA) Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) to a ```data``` folder in the project's directory or using DVC to download a zip file and unzip it (this will create the ```data``` folder in the project's directory).
```bash
dvc pull
unzip data.zip
```
Note: The DVC option might not be available due to the removal of the data from Google drive.

## Description
A CVAE using ViT in the encoder was designed, and a traditional fully convolution based CVAE was implemented. Initially, both the CNN-based CVAE and ViT-based CVAE generative models were trained. Then, the best performing model of each architecture evaluated on the test split was chosen. The chosen models were then used as the generator in a GAN structure and trained in an attempt to improve the visual quality of the generators. A pre-trained CNN-based classifier was fine-tuned on the training set as a baseline, additional classifiers, one for each generator, were fine-tuned with synthetic data from sampling the latent space distribution of the generators. The synthetic data was sampled in a manner such that each mini-batch had balanced classes, i.e. more images were generated of classes with low representation in the training
set. The performance of the classifiers were evaluated on overall and individual class accuracy to determine the suitability of Vision Transformers within generative modelling.

![network architecture](https://i.imgur.com/jeDVpqF.png)
## Results
