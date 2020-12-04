import nltk
import pandas
from nltk.metrics.distance import (
    edit_distance,
    jaccard_distance,
    )
from nltk.util import ngrams

import collections
from collections import Counter

import csv

from itertools import islice


with open('kbbi_data.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    list_kata = []
    for row in csv_reader:
        list_kata.append(row['katakunci'].strip())

spellings_series = pandas.Series(list_kata)

def exist(word):
    if word not in list_kata:
        return False
    else:
        return True

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))

def spell_check(word):
    n_grams = 2
    x = set(ngrams(word,n_grams))
    closest = {}
    spellings = spellings_series[spellings_series.str.startswith(word[0])] 
    for i in spellings:
        y = set(ngrams(i,n_grams))
        distance = jaccard_distance(x,y)
        closest[distance] = i 

    od = collections.OrderedDict(sorted(closest.items()))

    n_items = dict(take(3,od.items()))
    top3 = list(n_items.values())
    return top3

# print(jaccard('siapi',2))
# print(spellings_series[spellings_series.str.startswith('s')])