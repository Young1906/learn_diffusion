import torch
import einops 

from torch import nn
from torchsummary import summary
from torch.nn import functional as F

import lightning as L
import torchvision
import click
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score


from .cvae import LightTabularCVAE
from .utils import build_dataset

def load_tbcvae_decoder(
        input_dim:int,
        n_class:int,
        z_dim:int,
        pth: str) -> nn.Module:
    """
    Load a trained decoder for generation of synthetic samples
    """
    model = LightTabularCVAE.load_from_checkpoint(
            pth,
            input_dim=input_dim,
            n_class=n_class,
            z_dim=z_dim)

    model.eval()

    if torch.cuda.is_available():
        model.cuda()

    return model.cvae.decoder


def generate_synthesis_sample(
        decoder, counter, z_dim) -> list[np.ndarray]:
    """
    """
    # computer number of sample to generate for each class
    _max = np.array(list(counter.values())).max()

    n_samples = []

    for (c, n) in counter.items():
        n_samples.append((c, _max - n))


    # computer number of sample to generate for each class
    # generate additional data
    X_syn = []
    y_syn = []
    for (c, n) in n_samples:
        if n > 0:
            # random vector
            z = torch.randn(size=(n, z_dim))

            # label vector 
            _y = torch.ones(size=(n,)) * c
            _y = _y.type(torch.int64)
            
            _X_syn = decoder(z, _y)
            _X_syn = _X_syn.detach().clone().cpu().numpy()

            _y = _y.detach().clone().cpu().numpy()

            # Save the generated sample
            X_syn.append(_X_syn)
            y_syn.append(_y)
            

    X_syn = np.concatenate(X_syn, 0)
    y_syn = np.concatenate(y_syn)

    return X_syn, y_syn


@click.command()
@click.option("--checkpoint", type=str, required=True)
@click.option("--input_dim", type=int, required=True)
@click.option("--n_class", type=int, required=True)
@click.option("--z_dim", type=int, required=True)
@click.option("--dset", type=str, required=True)
def main(checkpoint:str,
         input_dim:int,
         n_class:int,
         z_dim:int,
         dset: str):

    # compute number of sample to be generate for each class
    X, y, counter = build_dataset(dset)

    # Train test split
    X, X_test, y, y_test = train_test_split(X, y, test_size=.25)

    # baseline
    # ---------------------------------------------------------
    print(X.shape, y.shape)
    clf = SVC()
    clf.fit(X, y)
    
    # Predict
    y_pred = clf.predict(X_test)
    f1_baseline = f1_score(y_test, y_pred)

    # CVAE
    # ---------------------------------------------------------
    # load trained decoder
    decoder = load_tbcvae_decoder(input_dim, n_class, z_dim, checkpoint) 
    
    # Generate synthesis data
    X_syn, y_syn = generate_synthesis_sample(decoder, counter, z_dim)

    # Training set
    X = np.concatenate([X, X_syn], 0)
    y = np.concatenate([y, y_syn])

    print(X.shape, y.shape)


    # Classifier & Train
    clf = SVC()
    clf.fit(X, y)

    # Predict
    y_pred = clf.predict(X_test)
    f1_cvae = f1_score(y_test, y_pred)

    with open("rs.txt", "a") as f:
        f.write(f"{f1_baseline:.5f}\t{f1_cvae}\n")



if __name__ == "__main__": main()
    # decoder = load_tbcvae_decoder(
    #         7, 8, 4, 
    #         "logs/ecoli/version_1/checkpoints/epoch=99-step=1700.ckpt")
