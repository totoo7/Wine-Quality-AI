import os
import numpy as np

from src.data_loader import load_red_wine, load_white_wine
from src.preprocessing import split_data, scale_data
from src.visualization import (
    plot_alcohol_vs_quality,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_correlation_heatmap
)
from src.models import train_linear_regression
from src.regression_evaluation import evaluate_regression

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

os.makedirs("graphs", exist_ok=True)

def run_pipeline(name, X, y):

    print(f"\n================ {name.upper()} ================")

    # VISUALIZATION
    plot_class_distribution(y, name)
    plot_alcohol_vs_quality(X, y, name)
    plot_correlation_heatmap(X, name)

    # SPLIT
    X_train, X_test, y_train, y_test = split_data(X, y)

    # SCALE (important for KNN)
    X_train, X_test, _ = scale_data(X_train, X_test)

    # KNN TUNING
    best_k = 1
    best_knn_acc = 0

    for k in range(1, 21):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)

        if acc > best_knn_acc:
            best_knn_acc = acc
            best_k = k

    knn_model = KNeighborsClassifier(n_neighbors=best_k)
    knn_model.fit(X_train, y_train)

    plot_confusion_matrix(knn_model, X_test, y_test, name + "_knn")

    # DECISION TREE TUNING
    best_depth = 1
    best_tree_acc = 0

    for depth in range(1, 16):
        model = DecisionTreeClassifier(max_depth=depth, random_state=1)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)

        if acc > best_tree_acc:
            best_tree_acc = acc
            best_depth = depth

    tree_model = DecisionTreeClassifier(max_depth=best_depth, random_state=1)
    tree_model.fit(X_train, y_train)

    plot_confusion_matrix(tree_model, X_test, y_test, name + "_tree")

    # FEATURE IMPORTANCE
    print("\nFeature Importance (Tree):")

    importances = tree_model.feature_importances_
    features = X.columns

    sorted_idx = np.argsort(importances)[::-1]

    print("\nTop 5 features:")
    for i in sorted_idx[:5]:
        print(f"{features[i]} -> {importances[i]:.4f}")

    # RESULTS
    print("\nRESULTS:")
    print(f"KNN best k={best_k} -> {best_knn_acc:.4f}")
    print(f"Tree best depth={best_depth} -> {best_tree_acc:.4f}")

    return {
        "knn": best_knn_acc,
        "tree": best_tree_acc,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }

def main():

    X_red, y_red = load_red_wine()
    X_white, y_white = load_white_wine()

    print("\nDATA LOADED")

    red_results = run_pipeline("red", X_red, y_red)
    white_results = run_pipeline("white", X_white, y_white)

    print("\n================ FINAL COMPARISON ================")

    print("\nRED WINE:")
    print(f"KNN: {red_results['knn']:.4f}")
    print(f"TREE: {red_results['tree']:.4f}")

    print("\nWHITE WINE:")
    print(f"KNN: {white_results['knn']:.4f}")
    print(f"TREE: {white_results['tree']:.4f}")

    best_score = max(
        red_results["knn"], red_results["tree"],
        white_results["knn"], white_results["tree"]
    )

    print("\nBEST OVERALL SCORE:", best_score)

    # REGRESSION (RED)
    print("\n================ REGRESSION (RED WINE) ================")

    red_linear = train_linear_regression(
        red_results["X_train"],
        red_results["y_train"]
    )

    r2_red, adj_r2_red = evaluate_regression(
        red_linear,
        red_results["X_test"],
        red_results["y_test"]
    )

    print("RED WINE:")
    print(f"R2: {r2_red:.4f}")
    print(f"Adjusted R2: {adj_r2_red:.4f}")

    # REGRESSION (WHITE)
    print("\n================ REGRESSION (WHITE WINE) ================")

    white_linear = train_linear_regression(
        white_results["X_train"],
        white_results["y_train"]
    )

    r2_white, adj_r2_white = evaluate_regression(
        white_linear,
        white_results["X_test"],
        white_results["y_test"]
    )

    print("WHITE WINE:")
    print(f"R2: {r2_white:.4f}")
    print(f"Adjusted R2: {adj_r2_white:.4f}")


if __name__ == "__main__":
    main()