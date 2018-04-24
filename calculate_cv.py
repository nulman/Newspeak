import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

import cfg


def calculate_cv(X, y):
    results = {
        'lr': [],
        # 'svm': [],
        # 'nb': [],
        # 'combined': []
    }
    lm = LogisticRegression()
    # svm = LinearSVC()
    # nb = MultinomialNB()
    # vc = VotingClassifier([('lm', lm), ('svm', svm), ('nb', nb)])

    return cross_val_score(lm, X, y, cv=10, scoring='accuracy').mean()
    # for category in cfg.categories_int:
    #     y_adj = np.array(y == category)
    #     results['lr'].append((cross_val_score(lm, X, y_adj, cv=10, scoring='accuracy').mean(), category))
        # results['svm'].append((cross_val_score(svm, X, y_adj, cv=10, scoring='accuracy').mean(), category))
        # results['nb'].append((cross_val_score(nb, X, y_adj, cv=10, scoring='accuracy').mean(), category))
        # results['combined'].append((cross_val_score(vc, X, y_adj, cv=10, scoring='accuracy').mean(), category))
    # return results


def get_lr(x, y):
    lm = LogisticRegression()
    lm_f = lm.fit(x, y)
    return lm_f
    # models = []
    # for c in cfg.categories_int:
    #     y_adj = np.array(y == c)
    #     lm = LogisticRegression()
    #     lm_f = lm.fit(x, y_adj)
    #     models.append(lm_f)
    # return models
