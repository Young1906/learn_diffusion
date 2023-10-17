import lightning as L
import click

from modules.data import EcoliDataModule, MNISTDataModule
from modules.cvae import LightTabularCVAE, LightCVAE

@click.command()
@click.option("--logname", type=str, required=True)
@click.option("--batch_size", type=int, required=True)
@click.option("--num_workers", type=int, required=True)
@click.option("--z_dim", type=int, required=True)
@click.option("--n_class", type=int, required=True)
@click.option("--max_epochs", type=int, required=True)
def main(logname: str, batch_size: int, num_workers: int, z_dim: int, n_class: int, max_epochs:int):
    dm = MNISTDataModule(batch_size, num_workers, "datasets")
    model = LightCVAE(z_dim, n_class)

    trainer = L.Trainer(
            logger = L.pytorch.loggers.tensorboard.TensorBoardLogger(
                "logs", name=logname),
            max_epochs = max_epochs)

    trainer.fit(model, dm)


if __name__ == "__main__": main()
