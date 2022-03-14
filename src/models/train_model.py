import logging

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
from src.data.make_dataset import CIFARDataModule, MNISTDataModule
from src.models.models import ViT, ViTVAE


def main(
    name: str = "test",
    model_type: str = "Classifier",
    max_epochs: int = 10,
    num_workers: int = 0,
    dim: int = 1024,
    depth: int = 12,
    heads: int = 16,
    mlp_dim: int = 2048,
):

    if model_type == "Classifier":
        model = ViT(
            image_size=32,
            patch_size=8,
            num_classes=10,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=0.5,
            emb_dropout=0.3,
        )
    if model_type == "ViTVAE":
        model = ViTVAE(
            image_size=32,
            patch_size=8,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
        )

    cifar = CIFARDataModule(batch_size=1024, num_workers=num_workers)
    cifar.prepare_data()
    cifar.setup()

    checkpoint_callback = ModelCheckpoint(
        dirpath=_PATH_MODELS + "/" + model_type,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        auto_insert_metric_name=True,
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=15, verbose=True, mode="min", strict=False
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    wandb_logger = WandbLogger(project="ViT-VAE", name=name)

    seed_everything(42, workers=True)

    trainer = Trainer(
        max_epochs=max_epochs,
        gpus=-1,
        precision=16,
        deterministic=True,
        default_root_dir=_PROJECT_ROOT,
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
        auto_lr_find=False,
        auto_scale_batch_size=False,
        auto_select_gpus=True,
        log_every_n_steps=25,
        logger=wandb_logger,
    )

    trainer.tune(model, datamodule=cifar)
    trainer.fit(model, datamodule=cifar)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
