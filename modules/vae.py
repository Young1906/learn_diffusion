import torch
import einops
import lightning as L
import torchvision

from torch import nn
from torchsummary import summary



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
    def __init__(self, z_dim: int):
        super().__init__()
        self.backbone = Backbone()
        self.gap = GAP()

        self.mu = nn.Sequential(
                nn.Linear(256, z_dim),
                nn.Tanh())

        self.logvar = nn.Sequential(
                nn.Linear(256, z_dim),
                nn.Tanh())


    def forward(self, x):
        # feature map
        fm = self.gap(self.backbone(x))

        # compute mu\_theta(x), sigma_\theta(x)
        mu = self.mu(fm)
        logvar = self.logvar(fm)

        return mu, logvar


class UpConv(nn.Module):
    def __init__(self, in_feat: int, out_feat: int, padding: int):
        super().__init__()
        self.seq = nn.Sequential(
                nn.ConvTranspose2d(
                    in_feat, out_feat, [2, 2], [2, 2],
                    output_padding=padding),
                nn.BatchNorm2d(out_feat),
                nn.ReLU())

    def forward(self, x):
        return self.seq(x)


class Reshape(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x):
        return einops.rearrange(x, 'b (c w h) -> b c w h', w=3, h=3)


class Decoder(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()
        self.seq = nn.Sequential(
                nn.Linear(z_dim, z_dim * 9),
                Reshape(),
                UpConv(z_dim, 128, 1),
                UpConv(128, 64, 0),
                UpConv(64, 1, 0),
                )

    def forward(self, x):
        return self.seq(x)


class VAE(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)


    def forward(self, x):
        mu, logvar = self.encoder(x)
        epsilon = torch.randn(mu.size()).type_as(x)

        z = mu + epsilon * torch.exp(logvar)
        x_res = self.decoder(z)

        return x_res, mu, logvar


class LightVAE(L.LightningModule):
    def __init__(self, z_dim: int):
        super().__init__()
        self.vae = VAE(z_dim)


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
        x_res, mu, logvar = self.vae(x)
        loss = self.elbo(x_res, x, mu, logvar)
        self.log("Loss", loss, prog_bar=True)


        return loss

    def on_train_epoch_end(self):
        # generate random image
        z = torch.randn((8, 256))
        imgs = self.vae.decoder(z)

        imgs = torchvision.utils.make_grid(imgs)
        self.logger.experiment.add_image("generated img", imgs,
                                         self.current_epoch)


    def configure_optimizers(self):
        return torch.optim.Adam(
                self.parameters(),
                lr=1e-2)



if __name__ == "__main__":
    net = VAE(256) 
    summary(net, (1, 28, 28), 256)

