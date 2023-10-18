from collections import defaultdict

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST, CelebA

class DataModuleWrapper(L.LightningDataModule):
    def __init__(
            self,
            dataset: Dataset,
            val_split: float,
            batch_size: int,
            num_workers: int):

        super().__init__()
        self.ds = dataset
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        N = len(self.full)
        vN = int(self.val_split * N)
        if stage == "fit":
            self.mnist_train, self.mnist_val = \
                    random_split(self.full, [N - vN, vN])

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            raise NotImplementedError("Not imeplemed yet")

        if stage == "predict":
            raise NotImplementedError("Not imeplemed yet")

    def train_dataloader(self):
        return DataLoader(
                self.mnist_train,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers = self.num_workers)

    def val_dataloader(self):
        return DataLoader(
                self.mnist_val,
                batch_size=self.batch_size,
                num_workers = self.num_workers)

    # def test_dataloader(self):
    #     return DataLoader(
    #             self.mnist_test,
    #             batch_size=self.batch_size,
    #             num_workers = self.num_workers)

    # def predict_dataloader(self):
    #     return DataLoader(
    #             self.mnist_predict,
    #             batch_size=self.batch_size)


def build_datamodule(
        name: str,
        val_split: float,
        batch_size: int,
        num_workers: int) -> L.LightningDataModule:
    """
    """

    ROOT = "datasets"

    if name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))

            ])

        ds = MNIST(
                root=ROOT,
                train=True,
                download=True,
                transform=transform)


    elif name == "celeba":
        transform = transforms.Compose([
            transforms.ToTensor(),])

        ds = CelebA(
                root=ROOT,
                split='all',
                download=True,
                transform=transform)
    else:
        raise ValueError(name)


    return DataModuleWrapper(
            dataset=ds,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers)



if __name__ == "__main__":
    dm = build_datamodule("celeba", .2, 64, 6)
