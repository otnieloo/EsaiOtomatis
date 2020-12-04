# import StemmerFactory class
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import re
import nltk
from nltk.tokenize import word_tokenize 
from nltk.corpus import words
from nltk.metrics.distance import (
    edit_distance,
    jaccard_distance,
    )
from nltk.util import ngrams
# nltk.download('words')
import numpy as np
import tesaurus
import kbbi
import pandas
import sklearn
import string
from sklearn.feature_extraction.text import TfidfVectorizer

kalimat1 = "Di sekolah kita belajar bahasa inggris dengan rkan dan guru tersayang 123 ! ! @$%"
kalimat2 = "Belajar bahasa inggris di sekolah dengan teman tercinta"

def preprocess(kalimat):

    # case folding
    kalimat = kalimat.lower()

    # number removal
    kalimat = re.sub(r"\d+", "", kalimat)

    # special char removal
    kalimat = kalimat.translate(str.maketrans("","",string.punctuation))

    # white space removal
    kalimat = kalimat.strip()

    # stop word sastrawi
    factory = StopWordRemoverFactory()
    stopword = factory.create_stop_word_remover()

    kalimat = stopword.remove(kalimat)
    # create stemmer & stemming process
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    output = stemmer.stem(kalimat)

    # tokenize process
    tokens = nltk.tokenize.word_tokenize(output)
    # tokens = nltk.tokenize.word_tokenize(kalimat)
    return tokens

def jaccard(s1,s2):
    s1 = set(s1)
    s2 = set(s2)
    i = s1.intersection(s2)
    return float(len(i))/(len(s1)+len(s2)-len(i))

def cosine_sim(s1,s2):
    s1 = set(s1)
    s2 = set(s2)
    rvector = s1.union(s2)
    l1 = []
    l2 = []
    c = 0
    for w in rvector: 
        if w in s1: l1.append(1) # create a vector 
        else: l1.append(0) 
        if w in s2: l2.append(1) 
        else: l2.append(0) 

    # cosine formula  
    # print(len(rvector))
    for i in range(len(rvector)): 
            c+= l1[i]*l2[i] 
    cosine = c / float((sum(l1)*sum(l2))**0.5) 
    return cosine 


# kalimat1_preprocess = preprocess(kalimat1)
# kalimat2_preprocess = preprocess(kalimat2)


def spell_check(kalimat1_preprocess):
    ne = {}
    for i in kalimat1_preprocess:
        if(not kbbi.exist(i)):
            spell_check = kbbi.spell_check(i)
            if spell_check != '':
                # index = kalimat1_preprocess.index(i)
                # kalimat1_preprocess[index] = spell_check[0]
                ne[i] = spell_check
    return ne

# print(len(ne))

# cosine = cosine_sim(kalimat1_preprocess,kalimat2_preprocess)
# jaccard_sim = jaccard(kalimat1_preprocess,kalimat2_preprocess)

def qe(kalimat1_preprocess,kalimat2_preprocess):

    unmatched_k1 = []
    for i in kalimat2_preprocess:
        if i not in kalimat1_preprocess:
            unmatched_k1.append(i)

    unmatched_k2 = []
    for i in kalimat1_preprocess:
        if i not in kalimat2_preprocess:
            unmatched_k2.append(i)

    sinonim_k2 = {}
    for i in unmatched_k2:
        sinonim_k2[i] = tesaurus.getSinonim(i)

    change = {}
    for i in sinonim_k2:
        for j in sinonim_k2[i]:
            if j in kalimat2_preprocess:
                change[j] = i
        # print(sinonim_k2[i])

    kalimat1_new = kalimat1_preprocess
    if change != '':
        for i in change:
            index = kalimat1_new.index(change[i])
            kalimat1_new[index] = i

    # if kalimat1_new == kalimat1_preprocess:
    #     return 'nc'
    # else:
    return kalimat1_new

# print(kalimat1_preprocess)
# print(kalimat2_preprocess)
# print(cosine)
# print(all)
# print(unmatched_k2)
# print(unmatched_k1)

# sinonim_k2 = {}
# for i in unmatched_k2:
#     sinonim_k2[i] = tesaurus.getSinonim(i)

# antonim_k2 = {}
# for i in unmatched_k2:
#     antonim_k2[i] = tesaurus.getAntonim(i)
#     # print(antonim_k2[i])

# change = {}
# for i in sinonim_k2:
#     for j in sinonim_k2[i]:
#         if j in kalimat2_preprocess:
#             change[j] = i
    # print(sinonim_k2[i])

# print(change)
# kalimat1_new = kalimat1_preprocess
# if change != '':
#     for i in change:
#         index = kalimat1_new.index(change[i])
#         kalimat1_new[index] = i

# cosine_new = cosine_sim(kalimat1_new,kalimat2_preprocess)
# print(cosine_new)
# print(kalimat1_new)