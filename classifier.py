import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split

import cfg


class classifier(object):

    # def __init__(self, common):
        # self._common = common
    @staticmethod
    def classify(tfidf_d, ratings, experiment='undefined'):
        # Prepare data for rating prediction
        # X_train, X_test, y_train, y_test = train_test_split(self._common.tfidf_d, self._common.data['star_rating'],
        #                                                     train_size=cfg.train_size,
        #                                                     test_size=cfg.test_size)

        cv = ShuffleSplit(n_splits=10, train_size=cfg.train_size, test_size=cfg.test_size, random_state=42)

        estimator = LogisticRegression()

        # Calculate model accuracies
        cv_scores = cross_val_score(estimator, tfidf_d, ratings, cv=cv,
                                    scoring='accuracy').mean()

        # Set the parameters by cross-validation
        print(f"Model accuracy predictions for {experiment}\n")
        print("(Score): {S:.1%}".format(S=cv_scores))
        print()
        return cv_scores

    @staticmethod
    def get_estimator():
        estimator = LogisticRegression(solver='lbfgs', random_state=42,
                                       multi_class='multinomial',
                                       class_weight='balanced')
        return estimator

    @classmethod
    def run_experiment(cls, X_train, X_test, Y_train , Y_test, experiment, classifier=None):
        if classifier is None:
            classifier = cls.get_estimator()
        # if not Y_train
        classifier.fit(X_train, Y_train)
        predictions = classifier.predict(X_test)
        Y_test = np.array(Y_test, dtype=np.int64)
        print(f"Model accuracy predictions for {experiment}\n")
        print("(Score): {S:.1%}".format(S=(sum(Y_test == predictions) / len(Y_test))))
        print('\n\n')
        return predictions



    # def plot_results(self):
    #     # Plot the results
    #     for cm in range(0, len(cfg.categories)):
    #         plot_coef('Top {N} words in ({R}) review model\nGreen = Associated | Red = Not Associated'.format(
    #             N=cfg.n_terms * 2,
    #             R=cfg.categories[cm]),
    #             self._common.lr_m[cm], self._common.tfidf_m.get_feature_names(), cfg.n_terms)
