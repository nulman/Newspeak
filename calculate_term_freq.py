from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

stemmer = SnowballStemmer("english")
tokenizer = RegexpTokenizer("[a-z']+")


def tokenize(text):
    tokens = tokenizer.tokenize(text)
    return [stemmer.stem(t) for t in tokens]


def get_tf(data, use_idf, max_df=1.0, min_df=1, ngram_range=(1, 1)):
    if use_idf:
        m = TfidfVectorizer(max_df=max_df, min_df=min_df, stop_words='english', ngram_range=ngram_range,
                            tokenizer=tokenize)
    else:
        m = CountVectorizer(max_df=max_df, min_df=min_df, stop_words='english', ngram_range=ngram_range,
                            tokenizer=tokenize)

    d = m.fit_transform(data)
    return m, d


