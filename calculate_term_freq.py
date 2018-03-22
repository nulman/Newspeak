from string import punctuation

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from Truecaser import Caser

stemmer = SnowballStemmer("english")
tokenizer = RegexpTokenizer("[\w']+")


def build_vocabulary():
    vocabulary = set()
    neg_pos_words = ['data\positive-words.txt', 'data\\negative-words.txt']
    for txt in neg_pos_words:
        with open(txt) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        vocabulary.update(content)
    return list(vocabulary)


def true_casing(tokens):
    caser = Caser()
    words = caser.getTrueCase([t for t in tokens])
    caser.unload()
    return words


def tokenize(text):
    # sentences = nltk.sent_tokenize(text)
    # for sentence in sentences:
    #     words = nltk.word_tokenize(sentence)
    #     tagged_words = nltk.pos_tag(words)
    #     ne_tagged_words = nltk.ne_chunk(tagged_words)
    tokens = tokenizer.tokenize(text)
    # tokens = pos_tag(tokens)
    # return [stemmer.stem(t) for t in tokens]
    return [w for w in tokens]


def get_tf(data, use_idf, max_df=1.0, min_df=1, ngram_range=(1, 1)):
    words = build_vocabulary()
    if use_idf:
        m = TfidfVectorizer(max_df=max_df, min_df=min_df, ngram_range=ngram_range, vocabulary=words + list(punctuation),
                            tokenizer=tokenize)
    else:
        m = CountVectorizer(max_df=max_df, min_df=min_df, stop_words='english', ngram_range=ngram_range,
                            tokenizer=tokenize)

    d = m.fit_transform(data.values.astype('U'))
    return m, d
