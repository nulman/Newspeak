from sklearn.cross_validation import train_test_split

import cfg
from calculate_cv import calculate_cv
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

    # def plot_results(self):
    #     # Plot the results
    #     for cm in range(0, len(cfg.categories)):
    #         plot_coef('Top {N} words in ({R}) review model\nGreen = Associated | Red = Not Associated'.format(
    #             N=cfg.n_terms * 2,
    #             R=cfg.categories[cm]),
    #             self._common.lr_m[cm], self._common.tfidf_m.get_feature_names(), cfg.n_terms)
