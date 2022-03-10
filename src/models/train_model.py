import logging
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import wandb


from src.models.models import VAE
from src.data.make_dataset import dataset, MNISTDataModule
from src import _PATH_DATA, _PATH_MODELS, _PROJECT_ROOT


def main():
    model = VAE()
    mnist = MNISTDataModule()
    mnist.setup()

    checkpoint_callback = ModelCheckpoint(dirpath=_PATH_MODELS, monitor="val_loss", mode="min", save_top_k= 3)
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=5, verbose=True, mode="min")


    trainer = Trainer(max_epochs=200,
                      gpus=1,
                      precision=16,
                      deterministic=True,
                      default_root_dir=_PROJECT_ROOT,
                      callbacks=[checkpoint_callback, early_stopping_callback],
                      auto_lr_find=True)

    trainer.tune(model, datamodule=mnist)
    trainer.fit(model, datamodule=mnist)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    seed_everything(42, workers=True)

    main()