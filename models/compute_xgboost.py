import sys
import os

code_dir = '### where the models and features codes are stored'
code_dir = '/home/minje/Projects/nlpfeatures'

sys.path.append(code_dir)
os.chdir(code_dir)
from features.word_features import ExtractWordFeatures
from features.readability_features import ExtractReadability
from features.lexicon_features import ExtractVader,ExtractLIWC,ExtractEmpath,ExtractHateSonar,ExtractAllLinguistics,ExtractNRC,ExtractNGrams
from features.embedding_features import ExtractEmbeddingSimilarities
from sklearn.pipeline import FeatureUnion
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from nltk.tokenize import TweetTokenizer
import os
from os.path import join


mode = 'all' # can be any of the modes below
dim = 'conflict' # can be any of the dimensions below

dims = ['social_support',
        'conflict',
        'trust',
        'fun',
        'similarity',
        'identity',
        'respect',
        'romance',
        'knowledge',
        'power']


tokenize = TweetTokenizer().tokenize


if mode=='readability':
    features = FeatureUnion([
        ('r1', ExtractReadability()),
    ])
elif mode=='vader':
    features = FeatureUnion([
        ('l1', ExtractVader()),
    ])
elif mode=='linguistics':
    features = FeatureUnion([
        ('l3', ExtractAllLinguistics()),
        ('w1', ExtractWordFeatures()),
    ])
elif mode == 'liwc':
    features = FeatureUnion([
        ('l6', ExtractLIWC()),
    ])
elif mode=='empath':
    features = FeatureUnion([
        ('l6', ExtractEmpath()),
    ])
elif mode=='hate':
    features = FeatureUnion([
        ('l10', ExtractHateSonar()),
    ])
elif mode=='ngrams':
    features = FeatureUnion([
        ('l12', ExtractNGrams(dimension=dim))
    ])
elif mode=='all':
    features = FeatureUnion([
        ('w1', ExtractWordFeatures()),
        ('r1', ExtractReadability()),
        ('l1', ExtractVader()),
        ('e1',ExtractEmbeddingSimilarities(dim=dim)),
        ('l3', ExtractAllLinguistics()),
        ('l4', ExtractLIWC()),
        ('l6', ExtractEmpath()),
        ('l10', ExtractHateSonar()),
        ('l12', ExtractNGrams(dimension=dim))
        ])
feature_names = features.get_feature_names()

# load model
model_dir = '### where the saved models are stored'
model_dir = '/10TBdrive/minje/models/xgboost/%s/%s'%(dim,mode)
with open(join(model_dir,'model.pckl'),'rb') as f:
    bst = pickle.load(f)


# replace this with any text data that you have
sentences = ['this is a sentence',
             'i hate what you do',
             'this is a good idea',
             'that is a bad idea',
             'i love that idea']

X = features.transform([tokenize(sent) for sent in sentences])
dtest = xgb.DMatrix(data=X,feature_names=feature_names,nthread=10)

y_prob = np.array(bst.predict(dtest))
y_pred = np.array(y_prob>=0.5,dtype=int)

for s,sent in zip(y_prob,sentences):
    print(round(s,3),sent)