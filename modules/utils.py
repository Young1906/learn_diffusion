import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import Counter



def _ecoli():
    pth = "datasets/ECOLI/ecoli.data"
    df = pd.read_csv(pth, sep="\s+", header=None)
    X = np.array(df.iloc[:, 1:-1].values)
    y = df.iloc[:, -1].values

    # label encoder
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    # Convert y from 8 class to binary class
    # {0: 143, 1: 77, 7: 52, 4: 35, 5: 20, 6: 5, 3: 2, 2: 2})
    y = (y >= 6) * 1 

    # Number of samples for each class
    return X, y, Counter(y)



def build_dataset(name: str):
    if name == "ecoli":
        return _ecoli()


    raise ValueError(name)


if __name__ == "__main__": 
    x, y, counter = _ecoli()
    print(counter)
