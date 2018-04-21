import sqlite3
from online_vectorizers import OnlineTfidfVectorizer
import calculate_term_freq as ctf
import pickle


def chunks(l, n=10000):
    prev = 1
    j = 1
    for i in range(n, l, n):
        yield [prev, i]
        prev = i + 1
        j = i
    if j != l:
        yield [prev, l]

def get_connection(path):
    return sqlite3.connect(path)

# chunk_query = r"select review_headline, review_body, star_rating from data where i >= {} and <= {};"
chunk_query = r'select (review_body || " " || review_headline) as text, ' \
              r'star_rating from data where i BETWEEN  {} and {} order by i;'

#ugly prototype to get evenly distributed ratings
'''select * from (select (review_body || " " || review_headline) as text, star_rating from data where star_rating = 1 limit 52000)
union
select * from (select (review_body || " " || review_headline) as text, star_rating from data where star_rating = 2 limit 52000)
union
select * from (select (review_body || " " || review_headline) as text, star_rating from data where star_rating = 3 limit 52000)
union
select * from (select (review_body || " " || review_headline) as text, star_rating from data where star_rating = 4 limit 52000)
union
select * from (select (review_body || " " || review_headline) as text, star_rating from data where star_rating = 5 limit 52000)'''

def get_table_size(con : sqlite3.Connection):
    curr = con.cursor()
    return curr.execute(r"select max(i) from data;").fetchone()[0]

class Vectorizer(object):
    def __init__(self, use_idf, max_df=1.0, min_df=1, ngram_range=(1, 1),):
        self.vectorizer = OnlineTfidfVectorizer(max_df=max_df, min_df=min_df, ngram_range=ngram_range,
                                      vocabulary=ctf.vocabulary, strip_accents='ascii', encoding='utf-8',
                                      tokenizer=ctf.tokenize)

    def fit(self, data):
        if hasattr(self.vectorizer, 'vocabulary_'):
            self.vectorizer.partial_refit(data)
        else:
            self.vectorizer.fit(data)
    def transform(self, data):
        return self.vectorizer.transform(data)

    def fit_transform(self, data):
        return self.vectorizer.fit_transform(data)

