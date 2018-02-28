import pandas as pd

from sklearn.model_selection import train_test_split

import cfg
import common
from calculate_cv import calculate_cv, get_lr
from cfg import n_topics
from compute_topics import get_lda, get_kmeans, show_topics, show_cluster_topics
from import_data import get_reviews
from calculate_term_freq import get_tf
from plot_data import get_svd, get_tsne, plot_scatter_2d, plot_coef


# Import review data
from test_data import test_review

data = get_reviews('data\Home_and_Kitchen_5.json', 6000)
pd.set_option('display.max_colwidth', -1)
data.sample(3)

# Calculate Term Frequencies
tf_m, tf_d = get_tf(data['reviewText'], use_idf=False, max_df=0.90, min_df=10)
tfidf_m, tfidf_d = get_tf(data['reviewText'], use_idf=True, max_df=0.90, min_df=10)

# Compute topics using Kmeans and LDA
# lda_m, lda_d = get_lda(tf_d, n_topics)
kmean_m, kmean_d = get_kmeans(tfidf_d, n_topics, scale=False)

# Show cluster top 15 words per topic
# print("Top 15 stemmed words per topic in LDA model\n")
# show_topics(lda_m, tf_m.get_feature_names(), 15)

# print("Top 15 stemmed words per cluster in Kmeans model\n")
# show_cluster_topics(kmean_d, tfidf_d, tfidf_m.get_feature_names(), 15)

# Prepare data for plotting
# svd_v, svd_m = get_svd(tfidf_d, 50)
# tnse_v, tsne_m = get_tsne(svd_m, 2, 25)

# lda_c = lda_d.argmax(axis=1)

# Plot Data
# matplotlib inline
# plot_scatter_2d(tsne_m[0], tsne_m[1], kmean_d, 1000, 'KMeans Clustering of Amazon Reviews using TFIDF (t-SNE Plot)')

# matplotlib inline
# plot_scatter_2d(tsne_m[0], tsne_m[1], lda_c, 1000, 'LDA Topics of Amazon Reviews using TF (t-SNE Plot)')

# Prepare data for rating prediction
X_train, X_test, y_train, y_test = train_test_split(tfidf_d, data['bucket'], test_size=0.3)

# Calculate model accuracies
cv_scores = calculate_cv(X_test, y_test)

print("Model accuracy predictions\n")
for m, s in cv_scores.items():
    for ss in s:
        print("{M} model ({R} rating): {S:.1%}".format(M=m.upper(), R=ss[1], S=ss[0]))
    print()

# Training the model of choice
# Assume we are happy with our logistic regression model
lr_m = get_lr(X_train, y_train)

# Plot the results

for c in range(0, len(cfg.cat)):
    plot_coef('Top {N} words in ({R}) review model\nGreen = Associated | Red = Not Associated'.format(N=cfg.n_terms * 2,
                                                                                                      R=cfg.cat[c]),
              lr_m[c], tfidf_m.get_feature_names(), cfg.n_terms)


c = common.common(tfidf_m, lr_m)
test_review(c, 'I bought these knives last week. I immediately returned these when they arrived damaged.')
test_review(c, 'This is the best toaster oven I have ever owned! I am glad I bought it.')