from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def calculate_cv(X, y):
    lm = LogisticRegression()
    return cross_val_score(lm, X, y, cv=10, scoring='accuracy').mean()
