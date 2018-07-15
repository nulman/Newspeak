# Amazon Sentiment Analysis Project

![Sentiment Analysis](https://cdn-images-1.medium.com/max/600/0*ga5rNPmVYBsCm-lz.)

This project is part of an NLP Lab course and its purpose:

1.	Learn Python’s NLP leading frameworks: SKLEARN and NLTK
2.	Develop a classifier based on Amazon reviews
3.	Try several features sets to achieve better model accuracy
4.	Understand the challenges in NLP sentiment analysis

# Prerequisites

1. python 3.6
2. sqlite3
3. csv
4. nltk
5. scikit-learn
6. matplotlib
7. numpy
8. pickle
9. true casing is based on [truecaser data](https://github.com/nreimers/truecaser/releases/download/v1.0/english_distributions.obj.zip)

# Generate SQL DB

We use SQLLite for our DBs, in order to create a new database from a TSV file, please run the following command:

usage: tosql.py [-h] path

positional arguments:
  path        Path to tsv file

optional arguments:

  -h, --help  show this help message and exit
  
# Run classifier
  
To run our project with database, please run the following command:
  
usage: main.py [-h] [--data1 DATA1] [--data2 DATA2] [--samples SAMPLES]
               [--classifier {logistic,linear}] [--domain {single,cross}]

optional arguments:
  -h, --help            show this help message and exit
  
  --data1 DATA1         Path to sqlite db representation of the data (default:
                        data\amazon_reviews_us_Watches_v1_00.db)
                        
  --data2 DATA2         Path to sqlite db representation of the data (default:
                        data\amazon_reviews_us_Books_v1_00.db)
                        
  --samples SAMPLES     samples PER CATEGORY per file (default: 100)
  
  --classifier {logistic,linear}
                        classifier to use (default: logistic)
                        
  --domain {single,cross}
                        run experiments on one db or between 2 dbs (default:
                        single)
                        
