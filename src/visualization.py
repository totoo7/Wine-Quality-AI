import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def plot_quality_distribution(y):
    plt.hist(y, bins=10)
    plt.title("Wine Quality Distribution")
    plt.xlabel("Quality")
    plt.ylabel("Count")
    plt.savefig("graphs/quality_distribution.png")
    plt.close()


def plot_alcohol_vs_quality(X, y):
    plt.scatter(X["alcohol"], y)
    plt.title("Alcohol vs Quality")
    plt.xlabel("Alcohol")
    plt.ylabel("Quality")
    plt.savefig("graphs/alcohol_vs_quality.png")
    plt.close()

def plot_class_distribution(y):
    import matplotlib.pyplot as plt
    import numpy as np

    unique, counts = np.unique(y, return_counts=True)

    plt.bar(unique, counts)
    plt.title("Class Distribution (Wine Quality)")
    plt.xlabel("Quality")
    plt.ylabel("Count")

    plt.savefig("graphs/class_distribution.png")
    plt.close()

def plot_confusion_matrix(model, X_test, y_test):
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title("Confusion Matrix")
    plt.savefig("graphs/confusion_matrix.png")
    plt.close()