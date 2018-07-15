

import csv, sqlite3
import argparse
import os

def tsv_to_sql(path, chunk_size=10000):
    # con = sqlite3.connect(r"data\amazon_reviews_us_Watches_v1_00.db")
    con = sqlite3.connect(os.path.join('data', os.path.basename(path).replace('.tsv', '.db')))
    cur = con.cursor()
    try:
        cur.execute("CREATE TABLE data (i int,marketplace,customer_id int,review_id,product_id,product_parent int,"
                    "product_title,product_category,star_rating int,helpful_votes int,total_votes int,"
                    "vine,verified_purchase,review_headline,review_body,"
                    "review_date datetime, primary key(i));") # use your column names here
    except sqlite3.OperationalError as e:
        cur.execute("delete from data")
        print(e)
        print('dumped table!')

    query = "INSERT INTO data (i,marketplace,customer_id,review_id,product_id,product_parent,product_title," \
            "product_category,star_rating,helpful_votes,total_votes,vine,verified_purchase,review_headline,review_body," \
            "review_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);"


    with open(path, encoding="utf8") as fin:
        # csv.DictReader uses first line in file for column headings by default
        dr = csv.DictReader(fin, delimiter="\t") # comma is default delimiter
        l = []
        i = 0
        j = 1
        for row in dr:
            r_data = tuple(row.values())
            if len(r_data) != 15:
                continue
            l.append([j] + list(row.values()))
            i += 1
            j += 1
            if i >= chunk_size:
                cur.executemany(query, l)
                i=0
                # print(l)
                l.clear()
                con.commit()
        if len(l) > 0:
            cur.executemany(query, l)
            con.commit()
    con.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Path to tsv file')

    args = parser.parse_args()
    tsv_to_sql(args.path)