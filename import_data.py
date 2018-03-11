import pandas as pd
import cfg
import json

def cat_y(y):
    if y <= 2.0:
        return cfg.categories[0]
    elif y >= 4.0:
        return cfg.categories[2]
    else:
        return cfg.categories[1]


def get_reviews(path, n_samples):
    dt = {}
    i = 0
    with open(path) as f:
        for d in f:
            dt[i] = json.loads(d)
            i += 1

    df = pd.DataFrame.from_dict(dt, orient='index')[['reviewText', 'overall']]
    df = df[df['reviewText'].apply(lambda x: len(x.split()) >= 45)]
    df['bucket'] = df['overall'].apply(cat_y)

    df = df.groupby('bucket').apply(lambda x: x.sample(n=n_samples))
    return df