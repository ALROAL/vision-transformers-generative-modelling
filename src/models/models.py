import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from pytorch_lightning import LightningModule
from vit_pytorch.deepvit import Transformer as Transformer_deep
from vit_pytorch.vit import Transformer, pair


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
    
    
class Discriminator_Patch_1(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator_Patch, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(in_channels = 512, out_channels = 1, kernel_size = 4, padding=1, bias=False))

    def forward(self, img_A):
        return self.model(img_A)
    

class Discriminator_Patch(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator_Patch, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(in_channels = 512, out_channels = 1, kernel_size = 4, padding=1, bias=False)
        )

    def forward(self, img_A):
        return self.model(img_A)
    
    
##############################
#          PatchGAN          #
##############################

class ViTVAE_PatchGAN(LightningModule):
    def __init__(
        self,
        image_size=(128,128),
        patch_size=16,
        num_classes = 4,
        dim=256,
        depth=4,
        heads=8,
        mlp_dim=256,
        channels=3,
        dim_head=64,
        ngf = 8,
        dropout=0.0,
        emb_dropout=0.0,
        landa = 100,
        kl_weight=1e-5,
        lr=1e-4,
        ):
        
        super().__init__()
        
        self.image_height, self.image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            self.image_height % patch_height == 0 and self.image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (self.image_height // patch_height) * (self.image_width // patch_width)
        patch_dim = (1+channels) * patch_height * patch_width
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)


        self.dim = dim
        self.mean_token = nn.Parameter(torch.randn(1, 1, dim))
        self.log_var_token = nn.Parameter(torch.randn(1, 1, dim))

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 2, dim))
        self.label_embedding = nn.Linear(num_classes, self.image_height*self.image_width)
        self.decoder_input = nn.Linear(dim + num_classes, dim)
        
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

        self.decoder_conv = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(dim, ngf * 16, (4, 4), (1, 1), bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf, channels, (4, 4), (2, 2), (1, 1), bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )
        # For now we will have a normal Discriminator; then I will change it to PatchGAN
        self.discriminator = Discriminator_Patch()
        self.landa =landa
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

    def decoder(self, z):
        result = self.decoder_input(z)
        result = rearrange(result, "b d -> b d 1 1")
        result = self.decoder_conv(result)
        return result
    
    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        z = eps.mul(std).add_(mean)
        # z = rearrange(z, "b d -> b 1 d")
        return z

    def forward(self, img, labels):
        # Labels 
        label_embeddings = self.label_embedding(labels)
        label_embeddings = label_embeddings.view(-1,self.image_height,self.image_width).unsqueeze(1)
        
        # Generator
        x = torch.cat([img,label_embeddings],dim = 1)
        x = self.encoder(x)

        # x += self.decoder_pos_embedding
        mean = x[:, 0]
        log_var = x[:, 1]
        

        z = self.reparameterize(mean, log_var)
        z = torch.cat([z, labels], dim = 1)
        out = self.decoder(z)

        # Discriminator
        real_label = self.discriminator(img)
        fake_label = self.discriminator(out)

        return out, img, mean, log_var, real_label, fake_label
    


    def sample(self, num_samples):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :return: (Tensor)
        """
        z = torch.randn(num_samples, 1, self.dim)
        labels = repeat(label, "d -> n d",n=num_samples)
        z = torch.cat([z, labels], dim = 1)
        samples = self.decoder(z)

        return samples

    def discriminator_loss(self, real_label, fake_label):
        # Loss with the real image
        loss_object = nn.CrossEntropyLoss()
        real_loss = loss_object(torch.ones_like(real_label),real_label)
        # Loss with the generated image
        generated_loss = loss_object(torch.zeros_like(fake_label), fake_label)

        total = real_loss + generated_loss

        return total

    def generator_loss(self,fake_label, out, img):
        # Want to make the answer of the discriminator all close to one
        loss_object = nn.CrossEntropyLoss()
        gan_loss = loss_object(torch.ones_like(fake_label), fake_label)
        #difference in image 
        loss_l1 = torch.mean(torch.absolute(img - out))
        total = gan_loss + (self.landa * loss_l1) 
        return total  

    def loss_function(self,recons_x, x, mu, log_var):
        """
        Computes the VAE loss function.
        """
        recons_loss = torch.sum(F.mse_loss(recons_x.view(recons_x.shape[0],-1), x.view(x.shape[0],-1),reduction="none"),dim=1)
        
        kld_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1)
    
        loss = torch.mean(recons_loss + kld_loss, dim=0)
        
        return {'loss': loss, 'Reconstruction_Loss':torch.mean(recons_loss.detach()), 'KLD':torch.mean(kld_loss.detach())}


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
        target = target.to(torch.float)

        recons_x, x, mu, log_var, real_label, fake_label = self(data, target)
        
        loss_Discriminator = self.discriminator_loss(real_label, fake_label)
        self.log("Discriminator_loss real image", loss_Discriminator)
        loss_Generator = self.generator_loss(fake_label, recons_x, x)
        self.log("Generator_loss real image", loss_Generator)

        loss_dict = self.loss_function(recons_x, x, mu, log_var)
        self.log_dict(loss_dict)


        return loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        data, target = batch
        target = target.to(torch.float)

        recons_x, x, mu, log_var, real_label, fake_label = self(data, target)

        loss_Discriminator = self.discriminator_loss(real_label, fake_label)
        self.log("GAN_loss validation real image", loss_Discriminator)
        loss_Generator = self.generator_loss(fake_label, recons_x, x)
        self.log("GAN_loss validation fake image", loss_Generator)

        loss_dict = self.loss_function(recons_x, x, mu, log_var)
        self.log_dict({
            'val_loss': loss_dict['loss'],
            'val_Reconstruction_Loss': loss_dict['Reconstruction_Loss'],
            'val_KLD': loss_dict['KLD']
        })

    def test_step(self, batch, batch_idx):
        data, target = batch
        target = target.to(torch.float)
        recons_x, x, mu, log_var, real_label, fake_label = self(data, target)

        loss_Discriminator = self.discriminator_loss(real_label, fake_label)
        self.log("GAN_loss test real image", loss_Discriminator)
        loss_Generator = self.generator_loss(fake_label, recons_x, x)
        self.log("GAN_loss test fake image", loss_Generator)

        loss_dict = self.loss_function(recons_x, x, mu, log_var)
        self.log_dict({
            'test_loss': loss_dict['loss'],
            'test_Reconstruction_Loss': loss_dict['Reconstruction_Loss'],
            'test_KLD': loss_dict['KLD']
        })

    def configure_optimizers(self):
        optimizer1 = optim.Adam(self.parameters(), lr=self.lr)
        optimizer2 = optim.Adam(self.discriminator.parameters(), lr = self.lr)
        lr_scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer1, patience=6)
        lr_scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer2,     patience=6)

        return ({
            "optimizer": optimizer1,
            "lr_scheduler": {
                "scheduler": lr_scheduler1,
                "monitor": "metric_to_track",},},
        {
            "optimizer": optimizer2, 
            "lr_scheduler": {
                "scheduler": lr_scheduler2,
                "monitor": "metric_to_track",
            },},)



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
        lr=3e-5,
        optim_choice="Adam"
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
        self.optim_choice = optim_choice

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
        if self.optim_choice == "Adam":
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
        if self.optim_choice == "SGD":
            optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
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
        lr=1e-4
    ):
        super().__init__()
        assert (
            image_size % patch_size == 0
        ), "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size**2
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size
            ),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer_deep(
            dim, depth, heads, dim_head, mlp_dim, dropout
        )

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))
        self.lr = lr
        self.save_hyperparameters()
        self.criterion = nn.CrossEntropyLoss()

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

    def loss(self, output, target):
        return self.criterion(output, target)

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
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
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
        image_size=(128, 128),
        patch_size=16,
        dim=256,
        depth=12,
        heads=16,
        mlp_dim=256,
        channels=3,
        dim_head=64,
        ngf=64,
        dropout=0.0,
        emb_dropout=0.0,
        kl_weight=1e-6,
        lr=5e-5,
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

        self.decoder_conv = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(dim, ngf * 16, (4, 4), (1, 1), bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf, channels, (4, 4), (2, 2), (1, 1), bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
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
        x = rearrange(x, "b d -> b d 1 1")

        x = self.dropout(x)

        x = self.decoder_conv(x)

        return x

    def reparameterize(self, mean, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        z = eps.mul(std).add_(mean)

        return z

    def forward(self, img):
        x = self.encoder(img)

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
        z = torch.randn(num_samples, self.dim)

        samples = self.decoder(z)

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



class ConvCVAE(LightningModule):
    def __init__(
        self,
        image_size=(128, 128),
        num_classes=4,
        dim=256,
        channels=3,
        ngf=8,
        lr=1e-4,
    ):
        super().__init__()
        self.image_height, self.image_width = pair(image_size)

        self.lr = lr
        self.save_hyperparameters()

        self.n_min = 1e-6
        self.n_max = 1e-3
        self.T_max = 1
        self.pi = np.pi
        self.first_epoch = True

        self.dim = dim
        self.mean = nn.Linear((ngf*16)*4*4, dim)
        self.log_var = nn.Linear((ngf*16)*4*4, dim)

        self.label_embedding = nn.Linear(num_classes, self.image_height*self.image_width)
        self.decoder_input = nn.Linear(dim + num_classes, dim)

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(channels + 1, out_channels=ngf, kernel_size= 3, stride= 2, padding = 1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(),
            
            nn.Conv2d(ngf, out_channels=ngf*2, kernel_size= 3, stride= 2, padding = 1),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(),
            
            nn.Conv2d(ngf*2, out_channels=ngf*4, kernel_size= 3, stride= 2, padding = 1),
            nn.BatchNorm2d(ngf*4),
            nn.LeakyReLU(),
            
            nn.Conv2d(ngf*4, out_channels=ngf*8, kernel_size= 3, stride= 2, padding = 1),
            nn.BatchNorm2d(ngf*8),
            nn.LeakyReLU(),

            nn.Conv2d(ngf*8, out_channels=ngf*16, kernel_size= 3, stride= 2, padding = 1),
            nn.BatchNorm2d(ngf*16),
            nn.LeakyReLU(),)

        self.decoder_conv = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(dim, ngf * 16, (4, 4), (1, 1), bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf, channels, (4, 4), (2, 2), (1, 1), bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def encoder(self, img):

        x = self.encoder_conv(img)

        return x

    def decoder(self, z):
        result = self.decoder_input(z)
        result = rearrange(result, "b d -> b d 1 1")
        result = self.decoder_conv(result)
        return result

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        z = eps.mul(std).add_(mean)

        return z

    def forward(self, img, labels):
        label_embeddings = self.label_embedding(labels)
        label_embeddings = label_embeddings.view(-1,self.image_height,self.image_width).unsqueeze(1)
        x = torch.cat([img,label_embeddings], dim = 1)
        x = self.encoder(x)

        x = torch.flatten(x, start_dim=1)

        mean = self.mean(x)
        log_var = self.log_var(x)

        z = self.reparameterize(mean, log_var)
        z = torch.cat([z, labels], dim = 1)
        x = self.decoder(z)

        return x, img, mean, log_var

    def sample(self, num_samples, label):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :return: (Tensor)
        """

        z = torch.randn(num_samples, self.dim)
        labels = repeat(label, "d -> n d",n=num_samples)
        z = torch.cat([z, labels], dim = 1)
        samples = self.decoder(z)
        return samples

    def loss_function(self,recons_x, x, mu, log_var):
        """
        Computes the VAE loss function.
        """
        if self.current_epoch == 1 & self.first_epoch:
            self.T_max = self.global_step
            self.first_epoch = False
            print(self.T_max)

        if self.current_epoch > 0:
            kl_weight = self.n_min+1/2*(self.n_max-self.n_min)*(1+np.cos(self.global_step/self.T_max*self.pi))
        else:
            kl_weight = 1e-4

        recons_loss =F.mse_loss(recons_x, x)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kl_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def training_step(self, batch, batch_idx):
        data, target = batch
        target = target.to(torch.float)
        recons_x, x, mu, log_var = self(data, target)
        loss_dict = self.loss_function(recons_x, x, mu, log_var)
        self.log_dict(loss_dict)
        return loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        data, target = batch
        target = target.to(torch.float)
        recons_x, x, mu, log_var = self(data, target)
        loss_dict = self.loss_function(recons_x, x, mu, log_var)
        self.log_dict({
            'val_loss': loss_dict['loss'],
            'val_Reconstruction_Loss': loss_dict['Reconstruction_Loss'],
            'val_KLD': loss_dict['KLD']
        })
    
    def test_step(self, batch, batch_idx):
        data, target = batch
        target = target.to(torch.float)
        recons_x, x, mu, log_var = self(data, target)
        loss_dict = self.loss_function(recons_x, x, mu, log_var)
        self.log_dict({
            'test_loss': loss_dict['loss'],
            'test_Reconstruction_Loss': loss_dict['Reconstruction_Loss'],
            'test_KLD': loss_dict['KLD']
        })


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10,factor=0.5)
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
            "strict": False,
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}


class CViTVAE(LightningModule):
    def __init__(
        self,
        image_size=(128, 128),
        patch_size=16,
        num_classes=4,
        dim=256,
        depth=4,
        heads=8,
        mlp_dim=256,
        channels=3,
        dim_head=64,
        ngf=8,
        dropout=0.0,
        emb_dropout=0.0,
        lr=1e-4,
    ):
        super().__init__()
        self.image_height, self.image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            self.image_height % patch_height == 0 and self.image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (self.image_height // patch_height) * (self.image_width // patch_width)
        patch_dim = (1+channels) * patch_height * patch_width

        self.lr = lr
        self.save_hyperparameters()


        self.dim = dim
        self.mean_token = nn.Parameter(torch.randn(1, 1, dim))
        self.log_var_token = nn.Parameter(torch.randn(1, 1, dim))

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 2, dim))

        self.label_embedding = nn.Linear(num_classes, self.image_height*self.image_width)
        self.decoder_input = nn.Linear(dim + num_classes, dim)

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.Linear(patch_dim, dim),
        )

        self.dropout = nn.Dropout(emb_dropout)

        self.encoder_transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout
        )

        self.decoder_conv = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(dim, ngf * 16, (4, 4), (1, 1), bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf, channels, (4, 4), (2, 2), (1, 1), bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

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

    def decoder(self, z):
        result = self.decoder_input(z)
        result = rearrange(result, "b d -> b d 1 1")
        result = self.decoder_conv(result)
        return result

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        z = eps.mul(std).add_(mean)

        return z

    def forward(self, img, labels):
        label_embeddings = self.label_embedding(labels)
        label_embeddings = label_embeddings.view(-1,self.image_height,self.image_width).unsqueeze(1)
        x = torch.cat([img,label_embeddings],dim = 1)
        x = self.encoder(x)

        mean = x[:, 0]
        log_var = x[:, 1]

        z = self.reparameterize(mean, log_var)
        z = torch.cat([z, labels], dim = 1)
        x = self.decoder(z)

        return x, img, mean, log_var

    def sample(self, num_samples, label):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :return: (Tensor)
        """

        z = torch.randn(num_samples, self.dim)
        labels = repeat(label, "d -> n d",n=num_samples)
        z = torch.cat([z, labels], dim = 1)
        samples = self.decoder(z)
        return samples

    def loss_function(self,recons_x, x, mu, log_var):
        """
        Computes the VAE loss function.
        """
        recons_loss = torch.sum(F.mse_loss(recons_x.view(recons_x.shape[0],-1), x.view(x.shape[0],-1),reduction="none"),dim=1)
        
        kld_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1)
    
        loss = torch.mean(recons_loss + kld_loss, dim=0)
        
        return {'loss': loss, 'Reconstruction_Loss':torch.mean(recons_loss.detach()), 'KLD':torch.mean(kld_loss.detach())}

    def training_step(self, batch, batch_idx):
        data, target = batch
        target = target.to(torch.float)
        recons_x, x, mu, log_var = self(data, target)
        loss_dict = self.loss_function(recons_x, x, mu, log_var)
        self.log_dict(loss_dict)
        return loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        data, target = batch
        target = target.to(torch.float)
        recons_x, x, mu, log_var = self(data, target)
        loss_dict = self.loss_function(recons_x, x, mu, log_var)
        self.log_dict({
            'val_loss': loss_dict['loss'],
            'val_Reconstruction_Loss': loss_dict['Reconstruction_Loss'],
            'val_KLD': loss_dict['KLD']
        })
    
    def test_step(self, batch, batch_idx):
        data, target = batch
        target = target.to(torch.float)
        recons_x, x, mu, log_var = self(data, target)
        loss_dict = self.loss_function(recons_x, x, mu, log_var)
        self.log_dict({
            'test_loss': loss_dict['loss'],
            'test_Reconstruction_Loss': loss_dict['Reconstruction_Loss'],
            'test_KLD': loss_dict['KLD']
        })


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5,factor=0.5)
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
            "strict": False,
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}



class Generator(nn.Module):
    def __init__(
        self,
        image_size=(128, 128),
        patch_size=16,
        num_classes=4,
        dim=256,
        depth=4,
        heads=8,
        mlp_dim=256,
        channels=3,
        dim_head=64,
        ngf=8,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.image_height, self.image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            self.image_height % patch_height == 0 and self.image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (self.image_height // patch_height) * (self.image_width // patch_width)
        patch_dim = (1+channels) * patch_height * patch_width

        self.mean_token = nn.Parameter(torch.randn(1, 1, dim))
        self.log_var_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 2, dim))
        self.label_embedding = nn.Linear(num_classes, self.image_height*self.image_width)
        self.decoder_input = nn.Linear(dim + num_classes, dim)
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.Linear(patch_dim, dim),
        )

        self.dropout = nn.Dropout(emb_dropout)

        self.encoder_transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout
        )

        self.decoder_conv = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(dim, ngf * 16, (4, 4), (1, 1), bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf, channels, (4, 4), (2, 2), (1, 1), bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

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

    def decoder(self, z):
        result = self.decoder_input(z)
        result = rearrange(result, "b d -> b d 1 1")
        result = self.decoder_conv(result)
        return result

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        z = eps.mul(std).add_(mean)

        return z

    def forward(self, img, labels):
        label_embeddings = self.label_embedding(labels)
        label_embeddings = label_embeddings.view(-1,self.image_height,self.image_width).unsqueeze(1)
        x = torch.cat([img,label_embeddings],dim = 1)
        x = self.encoder(x)

        mean = x[:, 0]
        log_var = x[:, 1]

        z = self.reparameterize(mean, log_var)
        z = torch.cat([z, labels], dim = 1)
        z = z.type_to(img)
        recons_img = self.decoder(z)

        return recons_img, mean, log_var


class CViTVAE_PatchGAN(LightningModule):
    def __init__(
        self,
        image_size=(128,128),
        patch_size=16,
        num_classes = 4,
        dim=256,
        depth=4,
        heads=8,
        mlp_dim=256,
        channels=3,
        dim_head=64,
        ngf = 8,
        dropout=0.0,
        emb_dropout=0.0,
        landa = 100,
        lr=1e-4
        ):
        
        super().__init__()
        

        self.generator = Generator(image_size=image_size,
                                   patch_size=patch_size,
                                   num_classes=num_classes,
                                   dim=dim,
                                   depth=depth,
                                   heads=heads,
                                   mlp_dim=mlp_dim,
                                   channels=channels,
                                   dim_head=dim_head,
                                   ngf=ngf,
                                   dropout=dropout,
                                   emb_dropout=emb_dropout)

        
        # For now we will have a normal Discriminator; then I will change it to PatchGAN
        self.discriminator = Discriminator_Patch()
        self.landa =landa
        self.lr = lr
        
        self.save_hyperparameters()


    def forward(self, img, labels):
        recons_img, mean, log_var = self.generator(img,labels)
    

        # # Discriminator
        real_label = self.discriminator(img)
        fake_label = self.discriminator(recons_img)

        return recons_img, mean, log_var

    def discriminator_loss(self, real_label, fake_label):
        # Loss with the real image
        loss_object = nn.CrossEntropyLoss()
        real_loss = loss_object(torch.ones_like(real_label),real_label)
        # Loss with the generated image
        generated_loss = loss_object(torch.zeros_like(fake_label), fake_label)

        total = real_loss + generated_loss

        return total

    def generator_loss(self,fake_label, out, img):
        # Want to make the answer of the discriminator all close to one
        loss_object = nn.CrossEntropyLoss()
        gan_loss = loss_object(torch.ones_like(fake_label), fake_label)
        #difference in image 
        loss_l1 = torch.mean(torch.absolute(img - out))
        total = gan_loss + (self.landa * loss_l1) 
        return total  

    def loss_function(self,recons_x, x, mu, log_var):
        """
        Computes the VAE loss function.
        """
        recons_loss = torch.sum(F.mse_loss(recons_x.view(recons_x.shape[0],-1), x.view(x.shape[0],-1),reduction="none"),dim=1)
        
        kld_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1)
    
        loss = torch.mean(recons_loss + kld_loss, dim=0)
        
        return {'loss': loss, 'Reconstruction_Loss':torch.mean(recons_loss.detach()), 'KLD':torch.mean(kld_loss.detach())}




    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, labels = batch
        labels = labels.to(torch.float)

        # train generator
        if optimizer_idx == 0:

            # generate images
            self.generated_imgs = self(imgs,labels)

            # log sampled images
            sample_imgs = self.generated_imgs[:6]
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image("generated_images", grid, 0)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
            tqdm_dict = {"g_loss": g_loss}
            output = OrderedDict({"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            # how well can it label as fake?
            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)

            fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict({"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

    def configure_optimizers(self):
        opt_g = optim.AdamW(self.generator.parameters(), lr=self.lr)
        opt_d = optim.AdamW(self.discriminator.parameters(),lr=self.lr)

        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt_g, patience=5,factor=0.5)
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
            "strict": False,
        }

        return {"optimizer": [opt_g,opt_d], "lr_scheduler": lr_scheduler_config}