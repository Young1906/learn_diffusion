import torch
import einops 

from torch import nn
from torchsummary import summary
from torch.nn import functional as F

import lightning as L
import torchvision



class ConvBlock(nn.Module):
    def __init__(self, in_feat:int, out_feat:int):
        super().__init__()
        self.conv = nn.Conv2d(in_feat, out_feat, [3, 3], [1, 1], "same")
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_feat)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.bn(out)
        return self.pool(out)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
                ConvBlock(1, 64),
                ConvBlock(64, 128),
                ConvBlock(128, 256),
        )

    def forward(self, x):
        return self.seq(x)


class GAP(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x):
        return x.mean((2, 3))


class Encoder(nn.Module):
    def __init__(self, z_dim: int, n_class: int):
        """
        """
        super().__init__()
        self.backbone = Backbone()
        self.gap = GAP()

        self.mu = nn.Sequential(
                nn.Linear(256 + n_class, z_dim),
                nn.Tanh())

        self.logvar = nn.Sequential(
                nn.Linear(256 + n_class, z_dim),
                nn.Tanh())

        self.n_class = n_class



    def forward(self, x, y):
        """
        y: class vector
        """
        # feature map
        fm = self.gap(self.backbone(x))
        y_onehot = F.one_hot(y, num_classes=self.n_class)

        # concat embedding of the image and class
        fm = torch.concat([fm, y_onehot], -1)

        # compute mu\_theta(x), sigma_\theta(x)
        mu = self.mu(fm)
        logvar = self.logvar(fm)

        return mu, logvar


class UpConv(nn.Module):
    def __init__(self, in_feat: int, out_feat: int, padding: int):
        super().__init__()
        self.seq = nn.Sequential(
                nn.Conv2d(
                    in_feat, out_feat,
                    [3, 3], [1, 1], "same"),

                nn.ConvTranspose2d(
                    out_feat, out_feat, [2, 2], [2, 2],
                    output_padding=padding),

                nn.BatchNorm2d(out_feat),

                nn.Conv2d(
                    out_feat, out_feat,
                    [1, 1], [1, 1], "same"),

                nn.Sigmoid())

    def forward(self, x):
        return self.seq(x)


class Reshape(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x):
        return einops.rearrange(x, 'b (c w h) -> b c w h', w=3, h=3)


class Decoder(nn.Module):
    def __init__(self, z_dim: int, n_class: int):
        super().__init__()
        self.seq = nn.Sequential(
                nn.Linear(z_dim + n_class, z_dim * 9),
                Reshape(),
                UpConv(z_dim, 128, 1),
                UpConv(128, 64, 0),
                UpConv(64, 1, 0),
                )
        self.n_class = n_class

    def forward(self, x, y):
        y_oh = F.one_hot(y, num_classes = self.n_class)
        return self.seq(torch.concat([x, y_oh], -1))


class CVAE(nn.Module):
    def __init__(self, z_dim: int, n_class: int):
        super().__init__()
        self.encoder = Encoder(z_dim, n_class)
        self.decoder = Decoder(z_dim, n_class)


    def forward(self, x, y):
        mu, logvar = self.encoder(x, y)
        epsilon = torch.randn(mu.size()).type_as(x)

        z = mu + epsilon * torch.exp(logvar)
        x_res = self.decoder(z, y)

        return x_res, mu, logvar


class LightCVAE(L.LightningModule):
    def __init__(self, z_dim: int, n_class: int):
        super().__init__()
        self.cvae = CVAE(z_dim, n_class)


    @staticmethod
    def elbo(x_res, x, mu, logvar) -> torch.Tensor:
        # reconstruction loss
        res = ((x_res - x)**2).mean()

        kl = -0.5 * torch.sum(
                1 + logvar - mu.pow(2) - logvar.exp())

        return 0.5 * res + 0.5 * kl


    def training_step(self, batch, batch_idx):
        # unpacking
        x, y = batch
        x_res, mu, logvar = self.cvae(x, y)
        loss = self.elbo(x_res, x, mu, logvar)

        self.log("Loss", loss, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        # generate random image
        z = torch.rand((8, 256)).to(self.device)
        c = torch.randint(low=0, high=10, size=(8,)).to(self.device)
        imgs = self.cvae.decoder(z, c)

        imgs = torchvision.utils.make_grid(imgs)
        self.logger.experiment.add_image("generated img", imgs,
                                         self.current_epoch)


    def configure_optimizers(self):
        return torch.optim.Adam(
                self.parameters(),
                lr=1e-2)


class DenseEcoder(nn.Module):
    def __init__(self, input_dim: int, n_class: int, z_dim: int):
        super().__init__()
        self.n_class = n_class
        self.fc = nn.Sequential(
                nn.Linear(input_dim, input_dim//2),
                nn.LeakyReLU(),
                nn.Linear(input_dim//2, input_dim//2),
                nn.Tanh())

        self.mu = nn.Sequential(
                nn.Linear(input_dim//2 + n_class, z_dim),
                nn.Tanh())

        self.logvar = nn.Sequential(
                nn.Linear(input_dim//2 + n_class, z_dim),
                nn.Tanh())

    def forward(self, x, y):
        y = F.one_hot(y, num_classes=self.n_class)
        z = torch.concat([self.fc(x), y], -1)
        return self.mu(z), self.logvar(z)


class DenseDecoder(nn.Module):
    def __init__(self, input_dim: int, n_class: int, z_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
                nn.Linear(z_dim + n_class, input_dim//2),
                nn.LeakyReLU(),
                nn.Linear(input_dim//2, input_dim),
                nn.Tanh())
        self.n_class=n_class

    def forward(self, x, y):
        y = F.one_hot(y, num_classes=self.n_class)
        return self.fc(torch.concat([x, y], -1))




class TabularCVAE(nn.Module):
    def __init__(self, input_dim: int, n_class: int, z_dim: int):
        super().__init__()
        self.encoder = DenseEcoder(input_dim, n_class, z_dim)
        self.decoder = DenseDecoder(input_dim, n_class, z_dim)

    def forward(self, x, y):
        mu, logvar  = self.encoder(x, y)
        epsilon = torch.randn(mu.size()).type_as(x)

        z = mu + epsilon * torch.exp(logvar)
        x_res = self.decoder(z, y)

        return x_res, mu, logvar



class LightTabularCVAE(L.LightningModule):
    def __init__(self, input_dim: int, n_class: int, z_dim: int):
        super().__init__()
        self.cvae = TabularCVAE(input_dim, n_class, z_dim)


    @staticmethod
    def elbo(x_res, x, mu, logvar) -> torch.Tensor:
        # reconstruction loss
        res = ((x_res - x)**2).mean()

        kl = -0.5 * torch.sum(
                1 + logvar - mu.pow(2) - logvar.exp())

        return 0.5 * res + 0.5 * kl


    def training_step(self, batch, batch_idx):
        # unpacking
        x, y = batch
        x_res, mu, logvar = self.cvae(x, y)
        loss = self.elbo(x_res, x, mu, logvar)

        self.log("train-loss", loss, prog_bar=True)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # unpacking
        x, y = batch
        x_res, mu, logvar = self.cvae(x, y)
        loss = self.elbo(x_res, x, mu, logvar)

        self.log("valid-loss", loss, prog_bar=True)

        return loss


    def configure_optimizers(self):
        return torch.optim.Adam(
                self.parameters(),
                lr=1e-3)

