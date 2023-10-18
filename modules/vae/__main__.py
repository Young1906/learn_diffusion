import click
import lightning as L
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

from modules.cvae import LightCVAE
from modules.data import build_datamodule 

@click.command()
@click.option("--logname", type=str, required=True)
@click.option("--dataset", type=str, required=True)
@click.option("--batch_size", type=int, required=True)
@click.option("--num_workers", type=int, required=True)
@click.option("--z_dim", type=int, required=True)
@click.option("--n_class", type=int, required=True)
@click.option("--max_epochs", type=int, required=True)
def main(
        logname: str,
        dataset: str,
        batch_size: int,
        num_workers: int,
        z_dim: int,
        n_class: int,
        max_epochs:int):

    # build data module
    # --------------------------------------------------
    dm = build_datamodule(
            name=dataset,
            batch_size=batch_size,
            num_workers=num_workers)

    # build model 
    # --------------------------------------------------
    model = LightCVAE(z_dim, n_class)

    # training 
    # --------------------------------------------------
    trainer = L.Trainer(
            logger = TensorBoardLogger("logs", name=logname),
            max_epochs = max_epochs)

    trainer.fit(model, dm)


if __name__ == "__main__": main()
