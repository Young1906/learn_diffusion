import lightning as L
import click

from modules.data import EcoliDataModule 
from modules.cvae import LightTabularCVAE 


@click.command()
@click.option("--logname", type=str, required=True)
@click.option("--batch_size", type=int, required=True)
@click.option("--num_workers", type=int, required=True)
@click.option("--pth", type=str, required=True) # pth to dataset
@click.option("--input_dim", type=int, required=True)
@click.option("--n_class", type=int, required=True)
@click.option("--z_dim", type=int, required=True)
@click.option("--max_epochs", type=int, required=True)
def main(
        logname:str,
        batch_size: int,
        num_workers: int,
        pth: str,
        input_dim: int,
        n_class: int,
        z_dim: int,
        max_epochs:int):

    # Fit a conditional VAE on the dataset
    dm = EcoliDataModule(batch_size, num_workers, pth)
    model = LightTabularCVAE(input_dim=input_dim, n_class=n_class, z_dim=z_dim)

    trainer = L.Trainer(
            logger = L.pytorch.loggers.tensorboard.TensorBoardLogger(
                "logs", name=logname),
            max_epochs = max_epochs)

    trainer.fit(model, dm)


if __name__ == "__main__": main()
