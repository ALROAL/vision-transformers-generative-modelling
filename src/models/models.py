import torch
import torch.utils.data
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from pytorch_lightning import LightningModule
from torch import nn, optim
from torch.nn import functional as F
from vit_pytorch.vit import Transformer
from vit_pytorch.deepvit import Transformer as Transformer_deep


class ViT(LightningModule):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        lr=3e-5
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

    def loss(self, y_hat, y):
        return F.cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = self.loss(output, target)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = self.loss(output, target)
        acc = (output.argmax(dim=1) == target).float().mean()
        self.log("val_acc", acc)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = self.loss(output, target)
        acc = (output.argmax(dim=1) == target).float().mean()
        self.log("test_acc", acc)
        self.log("test_loss", loss)
        return {"loss": loss, "acc": acc}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=6)
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
            "strict": True,
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

class DeepViT(LightningModule):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., lr=1e-4):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer_deep(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

    def loss(self, y_hat, y):
        return F.cross_entropy(y_hat.softmax(dim=1), y)

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = self.loss(output, target)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = self.loss(output, target)
        acc = (output.argmax(dim=1) == target).float().mean()
        self.log("val_acc", acc)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = self.loss(output, target)
        acc = (output.argmax(dim=1) == target).float().mean()
        self.log("test_acc", acc)
        self.log("test_loss", loss)
        return {"loss": loss, "acc": acc}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=6)
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
            "strict": True,
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}


class ViTVAE(LightningModule):
    def __init__(
        self,
        image_size,
        patch_size,
        dim,
        depth,
        heads,
        mlp_dim,
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        kl_weight=1e-5,
        lr=1e-3,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.dim = dim
        self.mean_token = nn.Parameter(torch.randn(1, 1, dim))
        self.log_var_token = nn.Parameter(torch.randn(1, 1, dim))

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 2, dim))

        self.decoder_pos_embedding = nn.Parameter(torch.randn(1, num_patches + 2, dim))

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.Linear(patch_dim, dim),
        )
        self.back_to_img = nn.Sequential(
            nn.Linear(dim, channels * image_height * image_width),
            Rearrange(
                "b 1 (p1 p2 c) -> b c p1 p2",
                c=channels,
                p1=image_height,
                p2=image_width,
            ),
        )

        self.dropout = nn.Dropout(emb_dropout)

        self.encoder_transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout
        )
        # For now we use the same transformer architecture for both the encoder and the decoder, but this could change.
        self.decoder_transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout
        )

        self.lr = lr
        self.kl_weight = kl_weight
        self.save_hyperparameters()

    def encoder(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        log_var_tokens = repeat(self.log_var_token, "() n d -> b n d", b=b)
        x = torch.cat((log_var_tokens, x), dim=1)

        mean_tokens = repeat(self.mean_token, "() n d -> b n d", b=b)
        x = torch.cat((mean_tokens, x), dim=1)

        x += self.pos_embedding

        x = self.dropout(x)

        x = self.encoder_transformer(x)

        return x

    def decoder(self, x):
        x = self.decoder_transformer(x)

        x = self.dropout(x)

        imgs = self.back_to_img(x)

        return imgs

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        z = eps.mul(std).add_(mean)
        z = rearrange(z, "b d -> b 1 d")
        return z

    def forward(self, img):
        x = self.encoder(img)
        x += self.decoder_pos_embedding
        mean = x[:, 0]
        log_var = x[:, 1]
        z = self.reparameterize(mean, log_var)
        out = self.decoder(z)

        return img, out, mean, log_var

    def sample(self, num_samples):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :return: (Tensor)
        """
        z = torch.randn(num_samples, 1, self.dim)

        samples = self.decode(z)
        return samples

    def elbo(self, recons_x, x, mu, logvar):
        """
        Computes the VAE loss function.
        """
        kl_divergence = 0.5 * torch.sum(-1 - logvar + mu.pow(2) + logvar.exp())
        MSE = nn.MSELoss(size_average=True)
        mse = MSE(recons_x, x)
        return mse + self.kl_weight * kl_divergence

    def training_step(self, batch, batch_idx):
        data, target = batch
        recons_x, x, mu, logvar = self(data)
        loss = self.elbo(recons_x, x, mu, logvar)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        recons_x, x, mu, logvar = self(data)
        loss = self.elbo(recons_x, x, mu, logvar)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        data, target = batch
        recons_x, x, mu, logvar = self(data)
        loss = self.elbo(recons_x, x, mu, logvar)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=6)
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
            "strict": False,
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
