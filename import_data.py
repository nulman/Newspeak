import pandas as pd
import cfg


def getDF(path):
    return pd.read_csv(path, sep='\t', header=0, error_bad_lines=False)


def cat_y(y):
    if y == 1.0:
        return cfg.categories[0]
    elif y == 2.0:
        return cfg.categories[1]
    elif y == 3.0:
        return cfg.categories[2]
    elif y == 4.0:
        return cfg.categories[3]
    else:
        return cfg.categories[4]


def get_reviews(path, n_samples):
    df = getDF(path)[['review_headline', 'review_body', 'star_rating']]
    # df = pd.DataFrame.from_dict(dt, orient='index')[['review_body', 'star_rating']]
    # df = df[df['review_body'].apply(lambda x: len(x.split()) >= 45)]
    df['bucket'] = df['star_rating'].apply(cat_y)

    df = df.groupby('bucket').apply(lambda x: x.sample(n=n_samples))
    return df


def get_average_per_user(path):
    df = getDF(path)
    df['freq'] = df.groupby('customer_id')['customer_id'].transform('count')
    df['sum'] = df.groupby('customer_id')['star_rating'].transform('sum')
    df['avg'] = df.apply(lambda row: row['sum'] / row['freq'], axis=1)

    new_df = df[['customer_id', 'star_rating', 'avg', 'freq', 'sum']].copy()
    return new_df
