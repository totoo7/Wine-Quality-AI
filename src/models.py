from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression

def train_decision_tree(X_train, y_train, max_depth=5):
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        criterion="entropy",
        random_state=1
    )
    model.fit(X_train, y_train)
    return model


def train_knn(X_train, y_train, k=5):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model