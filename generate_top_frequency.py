import do_things
import pandas as pd
import nltk
import pickle
from collections import defaultdict


# def top_frequent_words_in_text(database_path, n):
#     con = do_things.get_connection(database_path)
#
#     data = pd.read_sql_query(do_things.chunk_query, con)
#
#     sentences = []
#     tokens = []
#     for i in data['text']:
#         sentences.append(nltk.sent_tokenize(i))
#
#     for sentence in sentences:
#         for s in sentence:
#             tokens.append(nltk.word_tokenize(s))
#     tokens = [item for sublist in tokens for item in sublist]
#     tokens = [w.lower() for w in tokens if w.isalpha()]
#
#     stopwords = nltk.corpus.stopwords.words('english')
#     allWordExceptStopDist = nltk.FreqDist(w.lower() for w in tokens if w not in stopwords)
#     return allWordExceptStopDist.most_common(n)

def top_frequent_words_in_text(database_path, n=None):
    con = do_things.get_connection(database_path)
    cur = con.cursor()
    query = 'select (review_body || " " || review_headline) as text from data'
    res = cur.execute(query)
    words = defaultdict(int)
    stop_words = nltk.corpus.stopwords.words('english')
    stop_words = dict(zip(stop_words, range(len(stop_words))))
    for row in res:
        try:
            for word in (word.lower() for word in nltk.word_tokenize(row[0]) if word not in stop_words and word.isalpha()):
                words[word]+=1
        except:
            continue
    #here we remove any other words we dont want
    # for i in '!/><;&()#:-':
    #     del words[i]
    if 'br' in words: #this is a newline
        del words['br']
    if n is None:
        n = len(words)
    return nltk.FreqDist(words).most_common(n)


if __name__ == '__main__':
    top_words = top_frequent_words_in_text(r'data\\amazon_reviews_us_Books_v1_00.db', 10000)
    top_words = [a for a, b in top_words]

    output = open(f'top {10000} words books.pkl', 'wb')
    pickle.dump(top_words, output)
    output.close()
