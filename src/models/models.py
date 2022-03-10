import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from pytorch_lightning import LightningModule


class VAE(LightningModule):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)
        self.lr = 1e-2

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        x = x.view(-1, 784).float()
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), x, mu, logvar

    def elbo(self, recons_x, x, mu, logvar):
        """
        Computes the VAE loss function.
        """
        recons_loss = F.mse_loss(recons_x, x)

        KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)

        return recons_loss + KLD

    def sample(self, num_samples, current_device):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def training_step(self,batch,batch_idx):
        data, target = batch
        recons_x, x, mu, logvar = self(data)
        loss = self.elbo(recons_x, x, mu, logvar)
        self.log("train_loss", loss)
        return loss

    def validation_step(self,batch,batch_idx):
        data, target = batch
        recons_x, x, mu, logvar = self(data)
        loss = self.elbo(recons_x, x, mu, logvar)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(),lr=self.lr)