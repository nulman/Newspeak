import pandas as pd

from analyze_data import analyze_data
from calculate_term_freq import get_tf
from common import common
from import_data import get_reviews
from plot_data import plot_coef
from test_data import test_review
from classifier import classifier

# Import review data
data = get_reviews('data\Home_and_Kitchen_5.json', 6000)
pd.set_option('display.max_colwidth', -1)
data.sample(3)

# Init common class
cm = common(data)

# Calculate Term Frequencies
# tf_m, tf_d = get_tf(data['reviewText'], use_idf=False, max_df=0.90, min_df=10)
tfidf_m, tfidf_d = get_tf(data['reviewText'], use_idf=True, max_df=0.90, min_df=10)

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

# Test data
test_review(cm, 'I bought these knives last week. I immediately returned these when they arrived damaged.')
test_review(cm, 'This is the best toaster oven I have ever owned! I am glad I bought it.')