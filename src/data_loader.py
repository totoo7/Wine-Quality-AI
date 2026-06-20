import pandas as pd

def load_red_wine():
    df = pd.read_csv("data/winequality-red.csv", sep=";")
    X = df.drop("quality", axis=1)
    y = df["quality"]
    return X, y


def load_white_wine():
    df = pd.read_csv("data/winequality-white.csv", sep=";")
    X = df.drop("quality", axis=1)
    y = df["quality"]
    return X, y