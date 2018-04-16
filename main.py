import pandas as pd

from analyze_data import analyze_data
from calculate_term_freq import get_tf
from common import common
from import_data import get_reviews
from plot_data import plot_coef
from test_data import test_review
from classifier import classifier
import cfg

# Import review data
data = get_reviews('data\\amazon_reviews_us_Watches_v1_00.tsv',use_pickle=True, n=10000)#can do frac=0.x to get a fractional sample
pd.set_option('display.max_colwidth', -1)
# data.sample(cfg.categories.__len__())

# Init common class
cm = common(data)

# Calculate Term Frequencies
# tf_m, tf_d = get_tf(data['reviewText'], use_idf=False, max_df=0.90, min_df=10)

# We use individual words, bigrams and trigrams
tfidf_m, tfidf_d = get_tf(data['review_body'] + data['review_headline'], use_idf=True, max_df=cfg.max_df,
                          min_df=cfg.min_df, ngram_range=cfg.ngram_range)

# Propogate properties in common class
# cm.tf_m = tf_m
# cm.tf_d = tf_d
cm.tfidf_m = tfidf_m
cm.tfidf_d = tfidf_d

# Analyze data
# analyzer = analyze_data(cm)
# analyzer.analyze_kmeans()
# analyzer.analyze_lda()

# Classify data
classifier = classifier(cm)
classifier.classify()

# Plot results
classifier.plot_results()

first_star = data[data.star_rating == 1].iloc[0]
five_star = data[data.star_rating == 5].iloc[0]

# Test data
test_review(cm, first_star['review_headline'] + ' ' + first_star['review_body'])
test_review(cm, five_star['review_headline'] + ' ' + five_star['review_body'])
