import os
import numpy as np

from src.data_loader import load_wine_data
from src.preprocessing import split_data, scale_data
from src.visualization import (
    plot_quality_distribution,
    plot_alcohol_vs_quality,
    plot_class_distribution,
    plot_confusion_matrix
)
from src.correlation import plot_correlation_heatmap

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# =====================================================
# Create output folder for plots
# =====================================================
os.makedirs("graphs", exist_ok=True)


def main():

    # =====================================================
    # 1. LOAD DATA
    # =====================================================
    X, y = load_wine_data()

    print("\nDataset loaded successfully")
    print("Shape:", X.shape)


    # =====================================================
    # 2. VISUALIZATION (EDA)
    # =====================================================
    plot_quality_distribution(y)
    plot_class_distribution(y)
    plot_alcohol_vs_quality(X, y)

    # =====================================================
    # 2.1 CORRELATION HEATMAP (NEW)
    # =====================================================
    plot_correlation_heatmap(X)


    # =====================================================
    # 3. TRAIN / TEST SPLIT
    # =====================================================
    X_train, X_test, y_train, y_test = split_data(X, y)


    # =====================================================
    # 4. SCALING (important for KNN)
    # =====================================================
    X_train, X_test, _ = scale_data(X_train, X_test)


    # =====================================================
    # 5. KNN HYPERPARAMETER TUNING
    # =====================================================
    print("\n================ KNN TUNING ================")

    best_k = 1
    best_knn_acc = 0

    for k in range(1, 21):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)

        acc = knn.score(X_test, y_test)
        print(f"k={k} -> accuracy={acc:.4f}")

        if acc > best_knn_acc:
            best_knn_acc = acc
            best_k = k

    print("\nBEST K:", best_k)
    print("BEST KNN ACC:", best_knn_acc)

    # Final KNN model
    knn_model = KNeighborsClassifier(n_neighbors=best_k)
    knn_model.fit(X_train, y_train)

    plot_confusion_matrix(knn_model, X_test, y_test)


    # =====================================================
    # 6. DECISION TREE HYPERPARAMETER TUNING
    # =====================================================
    print("\n============= DECISION TREE TUNING =============")

    best_depth = 1
    best_tree_acc = 0

    for depth in range(1, 16):
        tree = DecisionTreeClassifier(
            max_depth=depth,
            criterion="entropy",
            random_state=42
        )

        tree.fit(X_train, y_train)
        acc = tree.score(X_test, y_test)

        print(f"depth={depth} -> accuracy={acc:.4f}")

        if acc > best_tree_acc:
            best_tree_acc = acc
            best_depth = depth

    print("\nBEST DEPTH:", best_depth)
    print("BEST TREE ACC:", best_tree_acc)

    # Final Decision Tree model
    tree_model = DecisionTreeClassifier(
        max_depth=best_depth,
        criterion="entropy",
        random_state=42
    )

    tree_model.fit(X_train, y_train)

    plot_confusion_matrix(tree_model, X_test, y_test)


    # =====================================================
    # 7. FEATURE IMPORTANCE (Feature Selection)
    # =====================================================
    print("\n============= FEATURE IMPORTANCE =============")

    importances = tree_model.feature_importances_
    features = X.columns

    sorted_idx = np.argsort(importances)[::-1]

    for i in sorted_idx:
        print(f"{features[i]}: {importances[i]:.4f}")

    print("\nTop 5 most important features:")
    for i in sorted_idx[:5]:
        print(f"{features[i]} -> {importances[i]:.4f}")


    # =====================================================
    # 8. FINAL COMPARISON
    # =====================================================
    print("\n================ FINAL COMPARISON ================")
    print(f"KNN Accuracy (best k={best_k}): {best_knn_acc:.4f}")
    print(f"Decision Tree Accuracy (best depth={best_depth}): {best_tree_acc:.4f}")

    if best_knn_acc > best_tree_acc:
        print("\nBEST MODEL: KNN")
    else:
        print("\nBEST MODEL: Decision Tree")


# =====================================================
# ENTRY POINT
# =====================================================
if __name__ == "__main__":
    main()