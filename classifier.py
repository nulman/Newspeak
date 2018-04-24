from sklearn.cross_validation import train_test_split

import cfg
from calculate_cv import calculate_cv, get_lr
# from plot_data import plot_coef


class classifier(object):

    def __init__(self, common):
        self._common = common

    def classify(self):
        # Prepare data for rating prediction
        X_train, X_test, y_train, y_test = train_test_split(self._common.tfidf_d, self._common.data['star_rating'],
                                                            train_size=cfg.train_size,
                                                            test_size=cfg.test_size)
        # Calculate model accuracies
        cv_scores = calculate_cv(X_test, y_test)

        print("Model accuracy predictions\n")
        print("(Score): {S:.1%}".format(S=cv_scores))
        print()
        # for m, s in cv_scores.items():
        #     for ss in s:
        #         print("{M} model ({R} rating): {S:.1%}".format(M=m.upper(), R=ss[1], S=ss[0]))
        #     print()

        # Training the model of choice
        # Assume we are happy with our logistic regression model
        lr_m = get_lr(X_train, y_train)
        self._common.lr_m = lr_m
    #
    # def plot_results(self):
    #     # Plot the results
    #     for cm in range(0, len(cfg.categories)):
    #         plot_coef('Top {N} words in ({R}) review model\nGreen = Associated | Red = Not Associated'.format(
    #             N=cfg.n_terms * 2,
    #             R=cfg.categories[cm]),
    #             self._common.lr_m[cm], self._common.tfidf_m.get_feature_names(), cfg.n_terms)
