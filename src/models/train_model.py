import logging
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb


from src.models.models import VAE, ViT
from src.data.make_dataset import dataset, MNISTDataModule, CIFARDataModule
from src import _PATH_DATA, _PATH_MODELS, _PROJECT_ROOT


def main():
    model = ViT(image_size=32,patch_size=8,num_classes=10,dim=128,depth=12,heads=8,mlp_dim=1024,dropout=0.1,emb_dropout=0.1)
    cifar = CIFARDataModule()
    cifar.setup()

    checkpoint_callback = ModelCheckpoint(dirpath=_PATH_MODELS, monitor="val_loss", mode="min", save_top_k= 3)
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=5, verbose=True, mode="min")

    wandb_logger = WandbLogger(project="ViT-VAE")

    trainer = Trainer(max_epochs=100,
                      gpus=-1,
                      precision=16,
                      deterministic=True,
                      default_root_dir=_PROJECT_ROOT,
                      callbacks=[checkpoint_callback, early_stopping_callback],
                      auto_lr_find=False,
                      auto_scale_batch_size=True,
                      auto_select_gpus=True,
                      logger=wandb_logger)

    trainer.tune(model, datamodule=cifar)
    trainer.fit(model, datamodule=cifar)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    seed_everything(42, workers=True)

    main()