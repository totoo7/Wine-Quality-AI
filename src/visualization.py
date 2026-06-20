import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

def _save(path):
    os.makedirs("graphs", exist_ok=True)
    plt.savefig(path)
    plt.close()

def plot_alcohol_vs_quality(X, y, name="wine"):
    plt.scatter(X["alcohol"], y)
    plt.title(f"Alcohol vs Quality ({name})")
    plt.xlabel("Alcohol")
    plt.ylabel("Quality")
    _save(f"graphs/{name}_alcohol_vs_quality.png")

def plot_class_distribution(y, name="wine"):
    unique, counts = np.unique(y, return_counts=True)

    plt.bar(unique, counts)
    plt.title(f"Class Distribution ({name})")
    plt.xlabel("Quality")
    plt.ylabel("Count")

    _save(f"graphs/{name}_class_distribution.png")

def plot_confusion_matrix(model, X_test, y_test, name="wine"):
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)

    plt.title(f"Confusion Matrix ({name})")

    _save(f"graphs/{name}_confusion_matrix.png")

def plot_correlation_heatmap(X, name):
    corr = X.corr().values
    features = X.columns

    fig, ax = plt.subplots(figsize=(10, 8))

    cax = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

    ax.set_xticks(np.arange(len(features)))
    ax.set_yticks(np.arange(len(features)))

    ax.set_xticklabels(features, rotation=90)
    ax.set_yticklabels(features)

    plt.colorbar(cax)

    plt.title("Feature Correlation Heatmap (Wine Quality)")
    plt.tight_layout()

    plt.savefig(f"graphs/{name}_correlation_heatmap.png")
    plt.close()