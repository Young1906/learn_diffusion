from collections import defaultdict

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST
from collections import Counter
from sklearn.preprocessing import LabelEncoder

class MNISTDataModule(L.LightningDataModule):
    def __init__(
            self,
            batch_size: int,
            num_workers: int,
            data_dir: str = "datasets" ):

        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

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

    def test_dataloader(self):
        return DataLoader(
                self.mnist_test,
                batch_size=self.batch_size,
                num_workers = self.num_workers)

    def predict_dataloader(self):
        return DataLoader(
                self.mnist_predict,
                batch_size=self.batch_size)


class EcoliDataset(torch.utils.data.Dataset):
    def __init__(self, pth: str):
        super().__init__()
        df = pd.read_csv(pth, sep="\s+", header=None)
        self.X = np.array(df.iloc[:, 1:-1].values)
        y = df.iloc[:, -1].values

        # label encoder
        self.le = LabelEncoder()
        self.le.fit(y)
        self.y = self.le.transform(y)
        
        # Convert y from 8 class to binary class
        # {0: 143, 1: 77, 7: 52, 4: 35, 5: 20, 6: 5, 3: 2, 2: 2})
        self.y = (self.y >= 6) * 1 


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        x, y = self.X[idx, :], self.y[idx]
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.int64)

        return x, y

    def __len__(self):
        N, _ = self.X.shape
        return N



class EcoliDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int, num_workers: int, pth: str):
        """
        Args:
        - batch size
        - num_workers
        - pth: path to datadir
        """
        # read .data files
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pth = pth

    def prepare_data(self):
        self.ecoli_full = EcoliDataset(self.pth)

    def setup(self, stage):
        if stage == "fit":
            N = len(self.ecoli_full)
            N_ = int(N * .2)
            self.ecoli_train, self.ecoli_valid = random_split(
                    self.ecoli_full, [N - N_, N_]) 


        if stage == "test":
            pass

        if stage == "predict":
            pass

    def train_dataloader(self):
        return DataLoader(
                self.ecoli_train,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers = self.num_workers)

    def val_dataloader(self):
        return DataLoader(
                self.ecoli_valid,
                batch_size=self.batch_size,
                num_workers = self.num_workers)

    



if __name__ == "__main__":
    dm = EcoliDataModule(8, 4, "datasets/ECOLI/ecoli.data")
    dm.prepare_data()
    dm.setup("fit")

    for (x, y) in dm.train_dataloader():
        print(x.shape, y)

