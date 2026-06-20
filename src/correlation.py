import numpy as np
import matplotlib.pyplot as plt


def plot_correlation_heatmap(X):
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

    plt.savefig("graphs/correlation_heatmap.png")
    plt.close()