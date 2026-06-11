# Wine Quality Prediction using Machine Learning

## Project Overview

This project is part of an Artificial Intelligence course and focuses on predicting the quality of wine based on physicochemical properties.

The project builds and compares machine learning models to classify wine quality using real-world data from the UCI Machine Learning Repository.

---

## Objective

The goal is to:

- Predict wine quality based on chemical features
- Analyze relationships between features
- Compare different machine learning models
- Perform hyperparameter tuning
- Apply feature importance analysis

---

## Dataset

The dataset contains physicochemical tests of wine samples such as:

- alcohol
- acidity
- pH
- sulphates
- chlorides
- residual sugar
- density

**Target variable:** wine quality (score from 3 to 9)

Source: UCI Wine Quality Dataset

---

## Machine Learning Models Used

### 1. K-Nearest Neighbors (KNN)
- Distance-based classification
- Hyperparameter tuned: `k (n_neighbors)`

### 2. Decision Tree Classifier
- Tree-based model
- Hyperparameter tuned: `max_depth`

---

## Workflow

The project follows a standard ML pipeline:

1. Load dataset
2. Exploratory Data Analysis (EDA)
3. Data visualization
4. Train-test split
5. Feature scaling
6. Model training
7. Hyperparameter tuning
8. Evaluation
9. Feature importance analysis
10. Model comparison

---

## Visualizations

The project includes:

- Wine quality distribution
- Alcohol vs Quality relationship
- Class distribution
- Correlation heatmap
- Confusion matrices

All graphs are saved in the `graphs/` directory.

---

## Feature Selection

Feature importance is extracted from the Decision Tree model to identify the most influential variables.

Top features typically include:

- alcohol
- volatile acidity
- sulphates
- density

---

## Model Evaluation

Models are evaluated using:

- Accuracy score
- Confusion matrix
- Cross-class classification performance

---

## Results

After hyperparameter tuning:

- **Best KNN Accuracy:** ~0.63
- **Best Decision Tree Accuracy:** ~0.60

Final selected model: **KNN**

---

## How to Run

### 1. Install dependencies

pip install numpy pandas matplotlib scikit-learn ucimlrepo

### 2. Run project
python main.py