from string import punctuation

import string
import pickle
import nltk
import numpy as np
from nltk.tag.perceptron import PerceptronTagger
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import cfg
from Truecaser import Caser

tagger = PerceptronTagger()

tokenizer = RegexpTokenizer("[\w']+")
# possible_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS',
#                  'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG',
#                  'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
possible_tags = list(set(tagger.tagdict.values()))
possible_cases = ['lower', 'upper', 'punctuation', 'title', 'mixed']

caser = None


# uncomment the following line to skip casing
# caser = type('',(),{'getTrueCase': lambda x: x})


def build_vocabulary():
    vocabulary = set()
    neg_pos_words = ['data\positive-words.txt', 'data\\negative-words.txt', 'data\\key-words.txt']
    for txt in neg_pos_words:
        with open(txt) as f:
            content = f.readlines()
        # content = ['_'.join([x.strip(), tag]) for x in content for tag in possible_tags]
        content = [x.strip() for x in content]
        vocabulary.update(content)
    return list(vocabulary)


def true_casing(tokens):
    caser = Caser()
    words = caser.getTrueCase([t for t in tokens])
    caser.unload()
    return words

def casing_tokenizer(text):
    tokens = []
    words = nltk.word_tokenize(text)
    for word in words:  # type: str
        if word.islower():
            tokens.append('lower')
        elif word.isupper():
            tokens.append('upper')
        elif len(word) == 1 and word in string.punctuation:
            tokens.append('punctuation')
        elif word.istitle():
            tokens.append('title')
        elif word.isalpha():
            tokens.append('mixed')
    return tokens

def tokenize(text):
    global caser
    if caser is None and tokenize.mode == 'pos':
        print('loading caser...', end='')
        caser = Caser()
        print('done.')
    sentences = nltk.sent_tokenize(text)
    tokens = []
    for sentence in sentences:
        if tokenize.mode == 'word':
            tokens.extend(nltk.word_tokenize(sentence))
        elif tokenize.mode == 'pos':
            tokens.extend([tag[1] for tag in tagger.tag(caser.getTrueCase(nltk.word_tokenize(sentence)))])
        elif tokenize.mode == 'case':
            tokens.extend(casing_tokenizer(sentence))
        elif tokenize.mode == 'count':
            return [str(len(nltk.word_tokenize(sentence)))]
    #     tagged_words = nltk.pos_tag(words)
    #     ne_tagged_words = nltk.ne_chunk(tagged_words)
    # tokens = tokenizer.tokenize(text)
    # cased_token = caser.getTrueCase(tokens)
    # cased_token = tagger.tag(cased_token)
    # cased_token = pos_tag(cased_token)
    # return [stemmer.stem(t) for t in tokens]
    return [t for t in tokens]
    # return ['_'.join([token, cased[1]]) for token, cased in zip(tokens, cased_token)]
tokenize.mode = 'word'

# Neg\Pos + Function words + Punctuation
# vocabulary = np.unique(build_vocabulary() + cfg.FUNCTION_WORDS  +
#                        list(punctuation)).tolist()

# Experiment #1 Only Negative and Positive words
# vocabulary = build_vocabulary()


# Experiment #2 Only Top 10000 frequent words


def get_vocabulary(variant='top'):
    if variant.lower() == 'sentiment':
        # print('build_vocabulary')
        return build_vocabulary()
    elif variant.lower() == 'top':
        # print(f'top {10000} words.pkl')
        with open(f'top {10000} words.pkl', 'rb') as pkl_file:
            return pickle.load(pkl_file)
    elif variant.lower() == 'pos':
        # print('possible_tags')
        return possible_tags
    elif variant.lower() == 'case':
        return possible_cases
    elif variant.lower() == 'count':
        return list(map(str,range(2001)))


def get_tf(data, use_idf, max_df=1.0, min_df=1, ngram_range=(1, 1)):
    if use_idf:
        m = TfidfVectorizer(max_df=max_df, min_df=min_df, ngram_range=ngram_range,
                            vocabulary=vocabulary, strip_accents='ascii', encoding='utf-8',
                            tokenizer=tokenize)
    else:
        m = CountVectorizer(max_df=max_df, min_df=min_df, stop_words='english', ngram_range=ngram_range,
                            tokenizer=tokenize)

    # data = data.values.astype('str')
    # data = np.core.defchararray.replace(data, '#$%', ' ')
    d = m.fit_transform(data)
    return m, d



