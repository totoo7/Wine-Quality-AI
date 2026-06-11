from ucimlrepo import fetch_ucirepo
import pandas as pd

def load_wine_data():
    dataset = fetch_ucirepo(id=186)

    X = dataset.data.features
    y = dataset.data.targets.squeeze()

    return X, y