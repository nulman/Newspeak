import time
# make sound!
import winsound

import pandas as pd

import cfg
import do_things
import calculate_term_freq as ctf
from classifier import linearClassifier, logisticClassifier
from common import common
from test_data import test_review
from scipy import sparse
import argparse


pd.set_option('display.max_colwidth', -1)

def single_mode(data, classify=True, classifier=logisticClassifier):
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
    # if classify: classifier.classify(top_pos_tfidf_d, data['star_rating'], 'top 10000 words + sentiment')

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


def multi_mode(data, data2, description='', classifier=logisticClassifier):
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

    # experiment #4
    top_pos_tfidf_d1 = sparse.hstack([top_tfidf_d1, pos_tfidf_d1])
    top_pos_tfidf_d2 = sparse.hstack([top_tfidf_d2, pos_tfidf_d2])
    classifier.run_experiment(top_pos_tfidf_d1, top_pos_tfidf_d2, data['star_rating'], data2['star_rating'],
                              'top 10000 words + pos')

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

if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data1', help='Path to sqlite db representation of the data',
                        default='data\\amazon_reviews_us_Watches_v1_00.db')
    parser.add_argument('--data2', help='Path to sqlite db representation of the data',
                        default='data\\amazon_reviews_us_Books_v1_00.db')
    parser.add_argument('--samples', help='samples PER CATEGORY per file', type=int, default=100)
    parser.add_argument('--classifier', help='classifier to use', default='logistic', choices=['logistic', 'linear'])
    parser.add_argument('--domain', help='run experiments on one db or between 2 dbs', default='single',
                        choices=['single', 'cross'])

    args = parser.parse_args()
    if args.classifier == 'logistic':
        classifier = logisticClassifier
    else:
        classifier = linearClassifier
    con = do_things.get_connection(args.data1)
    data = pd.read_sql_query(do_things.chunk_query(args.samples), con)
    if args.domain == 'single':
        top_tfidf_d, sentiment_tfidf_d, pos_tfidf_d, casing_tfidf_d, count_tfidf_d = single_mode(data=data,
                                                                                                 classify=True,
                                                                                                 classifier=classifier)
    else:
        con2 = do_things.get_connection('data\\amazon_reviews_us_Books_v1_00.db')
        data2 = pd.read_sql_query(do_things.chunk_query(args.samples), con2)
        multi_mode(data=data2, data2=data, description='*train on data2, test on data1*')
        multi_mode(data=data, data2=data2, description='*train on data1, test on data2*')
        multi_mode(data=pd.concat([data, data2]), data2=data2, description='*train on data1+data2, test on data2*')
        multi_mode(data=pd.concat([data, data2]), data2=data, description='*train on data1+data2, test on data1*')




    end = time.time()
    print('\nTiming:', end - start)
    winsound.Beep(100, 200)