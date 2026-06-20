from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def evaluate_regression(model, X_test, y_test):

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)

    n = X_test.shape[0]
    p = X_test.shape[1]

    adj_r2 = adjusted_r2(r2, n, p)

    return r2, adj_r2