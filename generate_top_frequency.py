import do_things
import pandas as pd
import nltk
import pickle


def top_frequent_words_in_text(database_path, n):
    con = do_things.get_connection(database_path)

    data = pd.read_sql_query(do_things.chunk_query, con)

    sentences = []
    tokens = []
    for i in data['text']:
        sentences.append(nltk.sent_tokenize(i))

    for sentence in sentences:
        for s in sentence:
            tokens.append(nltk.word_tokenize(s))
    tokens = [item for sublist in tokens for item in sublist]
    tokens = [w.lower() for w in tokens if w.isalpha()]

    stopwords = nltk.corpus.stopwords.words('english')
    allWordExceptStopDist = nltk.FreqDist(w.lower() for w in tokens if w not in stopwords)
    return allWordExceptStopDist.most_common(n)


if __name__ == '__main__':
    top_words = top_frequent_words_in_text(r'data\\amazon_reviews_us_Watches_v1_00.db', 10000)
    top_words = [a for a, b in top_words]

    output = open(f'top {10000} words.pkl', 'wb')
    pickle.dump(top_words, output)
    output.close()
