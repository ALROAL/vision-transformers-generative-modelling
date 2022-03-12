import logging

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import wandb
from src import _PATH_DATA, _PATH_MODELS, _PROJECT_ROOT
from src.data.make_dataset import CIFARDataModule, MNISTDataModule, dataset
from src.models.models import VAE, ViT


def main(name: str = "test", model_type: str = "Classifier", max_epochs: int = 10):
    if model_type == "Classifier":
        model = ViT(
            image_size=32,
            patch_size=8,
            num_classes=10,
            dim=128,
            depth=12,
            heads=8,
            mlp_dim=1024,
            dropout=0.1,
            emb_dropout=0.1,
        )

    cifar = CIFARDataModule(batch_size=512)
    cifar.prepare_data()
    cifar.setup()

    checkpoint_callback = ModelCheckpoint(
        dirpath=_PATH_MODELS, monitor="val_loss", mode="min", save_top_k=3
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=5, verbose=True, mode="min"
    )

    wandb_logger = WandbLogger(project="ViT-VAE", name=name)

    seed_everything(42, workers=True)

    trainer = Trainer(
        max_epochs=max_epochs,
        gpus=-1,
        precision=16,
        deterministic=True,
        default_root_dir=_PROJECT_ROOT,
        callbacks=[checkpoint_callback, early_stopping_callback],
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
