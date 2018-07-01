import time
# make sound!
import winsound

import pandas as pd

import cfg
import do_things
import calculate_term_freq as ctf
from classifier import classifier
from common import common
from test_data import test_review
from scipy import sparse
start = time.time()

# Import review data
# data = get_reviews('data\\amazon_reviews_us_Watches_v1_00.tsv',use_pickle=True, n=10000)#can do frac=0.x to get a fractional sample

pd.set_option('display.max_colwidth', -1)
# data.sample(cfg.categories.__len__())



con2 = do_things.get_connection('data\\amazon_reviews_us_Books_v1_00.db')
con = do_things.get_connection('data\\amazon_reviews_us_Watches_v1_00.db')
# n = do_things.get_table_size(con)
# n = 10000
# data = []
#change this if its too big for you
# chunk_size = n
#uncomment the following...
# for start, end in do_things.chunks(n, chunk_size):
#     print('starting: ', start, end)
#     frame = pd.read_sql_query(do_things.chunk_query.format(start, end), con)
#     print('loaded frame')
#     tfidf_m.fit(frame['text'])
#     data.append(frame)
#     print('finished: ',start, end)
# print('concatenating...')
# data = pd.concat(data)
# print('transforming...')
# tfidf_d = tfidf_m.transform(data['text'])

#...and comment these out
# data = pd.concat([pd.read_sql_query(do_things.chunk_query, con), pd.read_sql_query(do_things.chunk_query, con2)])
data = pd.read_sql_query(do_things.chunk_query, con)




def single_mode(classify=True):
    #experiment #1
    tfidf_m = do_things.Vectorizer(use_idf=True, max_df=cfg.max_df,
                              min_df=cfg.min_df, ngram_range=cfg.ngram_range, vocabulary_type='sentiment')
    sentiment_tfidf_d = tfidf_m.fit_transform(data['text'])
    if classify: classifier.classify(sentiment_tfidf_d, data['star_rating'], 'sentiment vocabulary')

    #experiment #2
    tfidf_m = do_things.Vectorizer(use_idf=True, max_df=cfg.max_df,
                              min_df=cfg.min_df, ngram_range=cfg.ngram_range, vocabulary_type='top')
    top_tfidf_d = tfidf_m.fit_transform(data['text'])
    if classify: classifier.classify(top_tfidf_d, data['star_rating'], 'top 10000 words')

    #experiment #3
    tfidf_m = do_things.Vectorizer(use_idf=True, max_df=cfg.max_df,
                              min_df=cfg.min_df, ngram_range=cfg.ngram_range, vocabulary_type='pos', tokenize_mode='pos')
    pos_tfidf_d = tfidf_m.fit_transform(data['text'])
    # classifier.classify(pos_tfidf_d, data['star_rating'], 'pos')

    #experiment #4
    top_pos_tfidf_d = sparse.hstack([top_tfidf_d, pos_tfidf_d])
    if classify: classifier.classify(top_pos_tfidf_d, data['star_rating'], 'top 10000 words + pos')

    #experiment #5
    # top_pos_tfidf_d = sparse.hstack([top_tfidf_d, sentiment_tfidf_d])
    # classifier.classify(top_pos_tfidf_d, data['star_rating'], 'top 10000 words + sentiment')

    #experiment #6
    top_pos_tfidf_d = sparse.hstack([top_tfidf_d, sentiment_tfidf_d, pos_tfidf_d])
    if classify: classifier.classify(top_pos_tfidf_d, data['star_rating'], 'top 10000 words + sentiment + pos')

    #experiment #7
    tfidf_m = do_things.Vectorizer(use_idf=True, max_df=cfg.max_df,
                              min_df=cfg.min_df, ngram_range=cfg.ngram_range, vocabulary_type='case', tokenize_mode='case',
                                   uncase=False)
    casing_tfidf_d = tfidf_m.fit_transform(data['text'])

    top_pos_tfidf_d = sparse.hstack([top_tfidf_d, sentiment_tfidf_d, pos_tfidf_d, casing_tfidf_d])
    if classify: classifier.classify(top_pos_tfidf_d, data['star_rating'], 'top 10000 words + sentiment + pos + casing')

    #experiment #8
    tfidf_m = do_things.Vectorizer(use_idf=True, max_df=cfg.max_df,
                              min_df=cfg.min_df, ngram_range=cfg.ngram_range, vocabulary_type='count')
    count_tfidf_d = tfidf_m.fit_transform(data['text'])
    if classify: classifier.classify(count_tfidf_d, data['star_rating'], 'word count')

    #experiment #9
    top_pos_tfidf_d = sparse.hstack([top_tfidf_d, sentiment_tfidf_d, pos_tfidf_d, casing_tfidf_d, count_tfidf_d])
    if classify: classifier.classify(top_pos_tfidf_d, data['star_rating'], 'top 10000 words + sentiment + pos + casing + word_count')
    return [top_tfidf_d, sentiment_tfidf_d, pos_tfidf_d, casing_tfidf_d, count_tfidf_d]

def multi_mode(data, data2, description=''):
    print(description)
    # experiment #1
    tfidf_m = do_things.Vectorizer(use_idf=True, max_df=cfg.max_df,
                                   min_df=cfg.min_df, ngram_range=cfg.ngram_range, vocabulary_type='sentiment')
    sentiment_tfidf_d1 = tfidf_m.fit_transform(data['text'])
    sentiment_tfidf_d2 = tfidf_m.fit_transform(data2['text'])
    classifier.run_experiment(sentiment_tfidf_d1, sentiment_tfidf_d2, data['star_rating'], data2['star_rating'],
                              'sentiment vocabulary')

    # experiment #2
    tfidf_m = do_things.Vectorizer(use_idf=True, max_df=cfg.max_df,
                                   min_df=cfg.min_df, ngram_range=cfg.ngram_range, vocabulary_type='top')
    top_tfidf_d1 = tfidf_m.fit_transform(data['text'])
    top_tfidf_d2= tfidf_m.fit_transform(data2['text'])
    classifier.run_experiment(top_tfidf_d1, top_tfidf_d2, data['star_rating'], data2['star_rating'], 'top 10000 words')

    # experiment #3
    tfidf_m = do_things.Vectorizer(use_idf=True, max_df=cfg.max_df,
                                   min_df=cfg.min_df, ngram_range=cfg.ngram_range, vocabulary_type='pos',
                                   tokenize_mode='pos')
    pos_tfidf_d1 = tfidf_m.fit_transform(data['text'])
    pos_tfidf_d2 = tfidf_m.fit_transform(data2['text'])
    # classifier.classify(pos_tfidf_d, data['star_rating'], 'pos')

    # experiment #4
    top_pos_tfidf_d1 = sparse.hstack([top_tfidf_d1, pos_tfidf_d1])
    top_pos_tfidf_d2 = sparse.hstack([top_tfidf_d2, pos_tfidf_d2])
    classifier.run_experiment(top_pos_tfidf_d1, top_pos_tfidf_d2, data['star_rating'], data2['star_rating'],
                              'top 10000 words + pos')

    # experiment #5
    # top_pos_tfidf_d = sparse.hstack([top_tfidf_d, sentiment_tfidf_d])
    # classifier.classify(top_pos_tfidf_d, data['star_rating'], 'top 10000 words + sentiment')

    # experiment #6
    top_pos_tfidf_d1 = sparse.hstack([top_tfidf_d1, sentiment_tfidf_d1, pos_tfidf_d1])
    top_pos_tfidf_d2 = sparse.hstack([top_tfidf_d2, sentiment_tfidf_d2, pos_tfidf_d2])
    classifier.run_experiment(top_pos_tfidf_d1, top_pos_tfidf_d2, data['star_rating'], data2['star_rating'],
                              'top 10000 words + sentiment + pos')

    # experiment #7
    tfidf_m = do_things.Vectorizer(use_idf=True, max_df=cfg.max_df,
                                   min_df=cfg.min_df, ngram_range=cfg.ngram_range, vocabulary_type='case',
                                   tokenize_mode='case',
                                   uncase=False)
    casing_tfidf_d1 = tfidf_m.fit_transform(data['text'])
    casing_tfidf_d2 = tfidf_m.fit_transform(data2['text'])

    top_pos_tfidf_d1 = sparse.hstack([top_tfidf_d1, sentiment_tfidf_d1, pos_tfidf_d1, casing_tfidf_d1])
    top_pos_tfidf_d2 = sparse.hstack([top_tfidf_d2, sentiment_tfidf_d2, pos_tfidf_d2, casing_tfidf_d2])
    classifier.run_experiment(top_pos_tfidf_d1, top_pos_tfidf_d2, data['star_rating'], data2['star_rating'],
                        'top 10000 words + sentiment + pos + casing')

    # experiment #8
    tfidf_m = do_things.Vectorizer(use_idf=True, max_df=cfg.max_df,
                                   min_df=cfg.min_df, ngram_range=cfg.ngram_range, vocabulary_type='count')
    count_tfidf_d1 = tfidf_m.fit_transform(data['text'])
    count_tfidf_d2 = tfidf_m.fit_transform(data2['text'])
    classifier.run_experiment(count_tfidf_d1, count_tfidf_d2, data['star_rating'], data2['star_rating'], 'word count')

    # experiment #9
    top_pos_tfidf_d1 = sparse.hstack([top_tfidf_d1, sentiment_tfidf_d1, pos_tfidf_d1, casing_tfidf_d1, count_tfidf_d1])
    top_pos_tfidf_d2 = sparse.hstack([top_tfidf_d2, sentiment_tfidf_d2, pos_tfidf_d2, casing_tfidf_d2, count_tfidf_d2])
    classifier.run_experiment(top_pos_tfidf_d1, top_pos_tfidf_d2, data['star_rating'], data2['star_rating'],
                        'top 10000 words + sentiment + pos + casing + word_count')

# data2 = pd.read_sql_query(do_things.chunk_query, con2)
# multi_mode(data=data2, data2=data, description='*train on books, test on watches*')
# multi_mode(data=data, data2=data2, description='*train on watches, test on books*')
# multi_mode(data=pd.concat([data, data2]), data2=data2, description='*train on watches+books, test on books*')
# multi_mode(data=pd.concat([data, data2]), data2=data, description='*train on watches+books, test on watches*')
top_tfidf_d, sentiment_tfidf_d, pos_tfidf_d, casing_tfidf_d, count_tfidf_d = single_mode()



# Init common class
# cm = common(data)

# Calculate Term Frequencies
# tf_m, tf_d = get_tf(data['reviewText'], use_idf=False, max_df=0.90, min_df=10)

# We use individual words, bigrams and trigrams
# tfidf_m, tfidf_d = get_tf(data['review_body'] + data['review_headline'], use_idf=True, max_df=cfg.max_df,
#                           min_df=cfg.min_df, ngram_range=cfg.ngram_range)

# Propogate properties in common class
# cm.tf_m = tf_m
# cm.tf_d = tf_d
# cm.tfidf_m = tfidf_m
# cm.tfidf_d = tfidf_d

# Analyze data
# analyzer = analyze_data(cm)
# analyzer.analyze_kmeans()
# analyzer.analyze_lda()

# Classify data
# classifier = classifier(cm)
# print('classifying...\n')
# classifier.classify()

# Plot results
# classifier.plot_results()

end = time.time()
print('\nTiming:', end - start)
winsound.Beep(100, 200)