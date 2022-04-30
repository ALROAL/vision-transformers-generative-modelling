import logging
import datetime

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

import wandb
from src import _PATH_DATA, _PATH_MODELS, _PROJECT_ROOT
from src.data.make_dataset import CelebADataModule, CIFARDataModule
from src.models.models import ViT, CViTVAE, ViTVAE, ConvCVAE, ViTVAE_GAN, ViTVAE_PatchGAN, ViTVAE_PatchGAN_prepared, ViTVAE_GAN_prepared, Classifier


def main(
    name: str = "test",
    model_type: str = "Classifier",
    max_epochs: int = 10,
    num_workers: int = 0,
    dim: int = 256,
    depth: int = 4,
    heads: int = 8,
    mlp_dim: int = 1024,
    lr: float = 1e-4,
    lr_discriminator: float = 1e-4,
    patch_size: int = 16,
    batch_size: int = 256,
    ngf: int = 8,
    kl_weight : int = 1e-5,
    frequency_generator: int = 1,
    frequency_discriminator:int = 1
):
    time = str(datetime.datetime.now())[:-10].replace(" ","-").replace(":","")

    # filename = "_".join(
    #     [
    #         str(p)
    #         for p in [
    #             model_type,
    #             patch_size,
    #             dim,
    #             depth,
    #             heads,
    #             mlp_dim,
    #             batch_size
    #         ]
    #     ]
    # )

    if model_type == "ViT":
        model = ViT(
            image_size=32,
            patch_size=patch_size,
            num_classes=6,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=0.5,
            emb_dropout=0.3,
            lr=lr,
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=_PATH_MODELS + "/" + model_type + time,
            filename='ViT-{epoch}',
            monitor="val_acc",
            mode="max",
            save_top_k=1,
            auto_insert_metric_name=True,
        )
        early_stopping_callback = EarlyStopping(
            monitor="val_acc", patience=30, verbose=True, mode="max", strict=False
        )


    if model_type == "ViTVAE":
        model = ViTVAE(
            image_size=(128, 128),
            patch_size=16,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=_PATH_MODELS + "/" + model_type + time,
            filename='ViTVAE-{epoch}',
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            auto_insert_metric_name=True,
        )
        early_stopping_callback = EarlyStopping(
            monitor="train_loss", patience=15, verbose=True, mode="min", strict=False
        )

    if model_type == "CViTVAE":
        time = str(datetime.datetime.now())[:-10].replace(" ","-").replace(":","")
        model = CViTVAE(
            image_size=(128, 128),
            patch_size=16,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            ngf=ngf,
            lr=lr
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=_PATH_MODELS + "/" + model_type + time,
            filename='CViTVAE-{epoch}',
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            auto_insert_metric_name=True,
        )
        early_stopping_callback = EarlyStopping(
            monitor="val_loss", patience=30, verbose=True, mode="min", strict=False
        )

    if model_type == "ConvCVAE":
        model = ConvCVAE(
            image_size=(128, 128),
            dim=dim,
            ngf=ngf,
            lr=lr
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=_PATH_MODELS + "/" + model_type + time,
            filename='ConvCVAE-{epoch}',
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            auto_insert_metric_name=True,
        )
        early_stopping_callback = EarlyStopping(
            monitor="train_loss", patience=15, verbose=True, mode="min", strict=False
        )
        
    if model_type == "ViTVAE_GAN":
        model = ViTVAE_GAN(
            image_size=(128, 128),
            dim=dim,
            frequency_generator = frequency_generator,
            frequency_discriminator = frequency_discriminator 
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=_PATH_MODELS + "/" + model_type + time,
            filename='ViTVAE_GAN-{epoch}',
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            auto_insert_metric_name=True,
        )
        early_stopping_callback = EarlyStopping(
            monitor="train_loss", patience=15, verbose=True, mode="min", strict=False
        )
        
    if model_type == "ViTVAE_PatchGAN":
        model = ViTVAE_PatchGAN(
            image_size=(128, 128),
            dim=dim,
            frequency_generator = frequency_generator,
            frequency_discriminator = frequency_discriminator 
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=_PATH_MODELS + "/" + model_type + time,
            filename='ViTVAE_PatchGAN-{epoch}',
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            auto_insert_metric_name=True,
        )
        early_stopping_callback = EarlyStopping(
            monitor="train_loss", patience=15, verbose=True, mode="min", strict=False
        )
        
    if model_type == "ViTVAE_PatchGAN_prepared":
        model = ViTVAE_PatchGAN_prepared(
            image_size=(128, 128),
            dim=dim,
            frequency_generator = frequency_generator,
            frequency_discriminator = frequency_discriminator 
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=_PATH_MODELS + "/" + model_type + time,
            filename='ViTVAE_PatchGAN_prepared-{epoch}',
            every_n_epochs = 25,
            auto_insert_metric_name=True,
            save_last=True
        )
        early_stopping_callback = EarlyStopping(
            monitor="train_loss", patience=1000, verbose=True, mode="min", strict=False
        )
        
    if model_type == "ViTVAE_GAN_prepared":
        model = ViTVAE_GAN_prepared(
            image_size=(128, 128),
            dim=dim,
            frequency_generator = frequency_generator,
            frequency_discriminator = frequency_discriminator 
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=_PATH_MODELS + "/" + model_type + time,
            filename='ViTVAE_GAN_prepared-{epoch}',
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            auto_insert_metric_name=True,
        )
        early_stopping_callback = EarlyStopping(
            monitor="train_loss", patience=15, verbose=True, mode="min", strict=False
        )

    if model_type == "Classifier":
        model = Classifier(lr=lr)
        checkpoint_callback = ModelCheckpoint(
            dirpath=_PATH_MODELS + "/" + model_type + time,
            filename='Classifier-{epoch}',
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            auto_insert_metric_name=True,
        )
        early_stopping_callback = EarlyStopping(
            monitor="val_loss", patience=20, verbose=True, mode="min", strict=False
        )

        celeb = CelebADataModule(batch_size=batch_size, num_workers=num_workers,classify=True)


    if model_type != "Classifier":
        celeb = CelebADataModule(batch_size=batch_size, num_workers=num_workers)

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    wandb_logger = WandbLogger(project="ViT-VAE", name=name)

    seed_everything(1234, workers=True)

    trainer = Trainer(
        max_epochs=max_epochs,
        gpus=-1,
        precision=16,
        deterministic=False,
        default_root_dir=_PROJECT_ROOT,
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
        auto_lr_find=False,
        auto_scale_batch_size=False,
        auto_select_gpus=True,
        log_every_n_steps=25,
        logger=wandb_logger,
    )

    # trainer.tune(model, datamodule=celeb)
    trainer.fit(model, datamodule=celeb)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
