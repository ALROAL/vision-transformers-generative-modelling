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
from src.data.make_dataset import CelebADataModule, CIFARDataModule
from src.models.models import DeepViT, ViT, ViTCVAE_A, ViTCVAE_R, ViTVAE


def main(
    name: str = "test",
    model_type: str = "ViT",
    max_epochs: int = 10,
    num_workers: int = 0,
    dim: int = 1024,
    depth: int = 4,
    heads: int = 8,
    mlp_dim: int = 1024,
    kl_weight: float = 1e-5,
    lr: float = 3e-5,
    patch_size: int = 16,
    batch_size: int = 256,
    optim_choice: str = "Adam",
    ngf: int = 8,
):
    filename = "_".join(
        [
            str(p)
            for p in [
                model_type,
                patch_size,
                dim,
                depth,
                heads,
                mlp_dim,
                batch_size,
                kl_weight,
            ]
        ]
    )

    if model_type == "ViT":
        model = ViT(
            image_size=32,
            patch_size=patch_size,
            num_classes=10,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=0.5,
            emb_dropout=0.3,
            lr=lr,
            optim_choice=optim_choice,
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=_PATH_MODELS + "/" + model_type,
            filename=filename,
            monitor="val_acc",
            mode="max",
            save_top_k=1,
            auto_insert_metric_name=True,
        )
        early_stopping_callback = EarlyStopping(
            monitor="val_acc", patience=30, verbose=True, mode="max", strict=False
        )

    if model_type == "DeepViT":
        model = DeepViT(
            image_size=32,
            patch_size=patch_size,
            num_classes=10,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=0.1,
            emb_dropout=0.0,
            lr=lr,
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=_PATH_MODELS + "/" + model_type,
            filename=filename,
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
            kl_weight=kl_weight,
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=_PATH_MODELS + "/" + model_type,
            filename=filename,
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            auto_insert_metric_name=True,
        )
        early_stopping_callback = EarlyStopping(
            monitor="train_loss", patience=15, verbose=True, mode="min", strict=False
        )

    if model_type == "ViTCVAE_R":
        model = ViTCVAE_R(
            image_size=(128, 128),
            patch_size=16,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            ngf=ngf,
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=_PATH_MODELS + "/" + model_type,
            filename='{epoch}-{val_loss:3f}',
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            auto_insert_metric_name=True,
        )
        early_stopping_callback = EarlyStopping(
            monitor="val_loss", patience=5, verbose=True, mode="min", strict=False
        )

    if model_type == "ViTCVAE_A":
        model = ViTCVAE_A(
            image_size=(128, 128),
            patch_size=16,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=_PATH_MODELS + "/" + model_type,
            filename=filename,
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            auto_insert_metric_name=True,
        )
        early_stopping_callback = EarlyStopping(
            monitor="train_loss", patience=15, verbose=True, mode="min", strict=False
        )

    celeb = CelebADataModule(batch_size=batch_size, num_workers=num_workers)
    celeb.setup()

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    wandb_logger = WandbLogger(project="ViT-VAE", name=name)

    seed_everything(42, workers=True)

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

    trainer.tune(model, datamodule=celeb)
    trainer.fit(model, datamodule=celeb)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
