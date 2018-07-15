import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score, cross_val_predict, learning_curve
import matplotlib.pyplot as plt


import cfg


class linearClassifier(object):

    @staticmethod
    def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                            n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
        """
        Generate a simple plot of the test and training learning curve.

        Parameters
        ----------
        estimator : object type that implements the "fit" and "predict" methods
            An object of that type which is cloned for each validation.

        title : string
            Title for the chart.

        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        ylim : tuple, shape (ymin, ymax), optional
            Defines minimum and maximum yvalues plotted.

        cv : int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:
              - None, to use the default 3-fold cross-validation,
              - integer, to specify the number of folds.
              - An object to be used as a cross-validation generator.
              - An iterable yielding train/test splits.

            For integer/None inputs, if ``y`` is binary or multiclass,
            :param train_sizes:
            :class:`StratifiedKFold` used. If the estimator is not a classifier
            or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validators that can be used here.

        n_jobs : integer, optional
            Number of jobs to run in parallel (default 1).
        """
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")

        plt.legend(loc="best")
        return plt

    @staticmethod
    def classify(tfidf_d, ratings, experiment='undefined'):
        cv = ShuffleSplit(n_splits=10, train_size=cfg.train_size, test_size=cfg.test_size, random_state=42)

        estimator = LinearRegression()

        # Calculate model accuracies
        scoring = 'neg_mean_squared_error'
        cv_scores = -1 * cross_val_score(estimator, tfidf_d, ratings, cv=cv, scoring=scoring).mean()
        # cv_scores = []
        predicted = cross_val_predict(estimator, tfidf_d, ratings, cv=10)

        # uncomment the following block to see scatter plots
        # fig, ax = plt.subplots()
        # ax.scatter(ratings, predicted, edgecolors=(0, 0, 0))
        # ax.plot([ratings.min(), ratings.max()], [ratings.min(), ratings.max()], 'k--', lw=4)
        # ax.set_xlabel('Measured')
        # ax.set_ylabel('Predicted')
        # plt.title("Model accuracy predictions for {S}".format(S=experiment))
        # plt.show()

        # Set the parameters by cross-validation
        print(f"Model accuracy predictions for {experiment}\n")
        # print("(Score): {S:.1%}".format(S=cv_scores))
        print("(Score): {S}".format(S=cv_scores))
        print()

        return cv_scores

class logisticClassifier(object):


    @staticmethod
    def classify(tfidf_d, ratings, experiment='undefined'):
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
