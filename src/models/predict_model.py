import torch

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger

from src import _PATH_DATA, _PATH_MODELS, _PROJECT_ROOT
from src.data.make_dataset import CelebADataModule
from src.models.models import  CViTVAE, ConvCVAE


wandb_logger = WandbLogger(project="ViT-VAE", name=Testing)

seed_everything(1234, workers=True)

celeb = CelebADataModule(batch_size=1024, num_workers=0)
celeb.setup()

trainer = Trainer(logger=wandb_logger)

model = CViTVAE.load_from_checkpoint(_PATH_MODELS+"\CViTVAE2022-04-29-1423\CViTVAE-epoch=170.ckpt")
model.eval()
trainer.test(model,datamodule=celeb)

model = CViTVAE.load_from_checkpoint(_PATH_MODELS+"\CViTVAE2022-04-29-1606\CViTVAE-epoch=133.ckpt")
model.eval()
trainer.test(model,datamodule=celeb)

model = CViTVAE.load_from_checkpoint(_PATH_MODELS+"\CViTVAE2022-04-29-1735\CViTVAE-epoch=174.ckpt")
model.eval()
trainer.test(model,datamodule=celeb)


model = ConvCVAE.load_from_checkpoint(_PATH_MODELS+"\ConvCVAE2022-04-29-2156\ConvCVAE-epoch=638.ckpt")
model.eval()
trainer.test(model,datamodule=celeb)

model = ConvCVAE.load_from_checkpoint(_PATH_MODELS+"\ConvCVAE2022-04-30-0830\ConvCVAE-epoch=619.ckpt")
model.eval()
trainer.test(model,datamodule=celeb)

model = ConvCVAE.load_from_checkpoint(_PATH_MODELS+"\ConvCVAE2022-04-30-0830\ConvCVAE-epoch=792.ckpt")
model.eval()
trainer.test(model,datamodule=celeb)

model = ConvCVAE.load_from_checkpoint(_PATH_MODELS+"\ConvCVAE2022-04-30-1854\ConvCVAE-epoch=349.ckpt")
model.eval()
trainer.test(model,datamodule=celeb)

model = ConvCVAE.load_from_checkpoint(_PATH_MODELS+"\ConvCVAE2022-05-01-2202\ConvCVAE-epoch=196.ckpt")
model.eval()
trainer.test(model,datamodule=celeb)