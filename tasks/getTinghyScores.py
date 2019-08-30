import os
from os.path import join
import re
from nltk import sent_tokenize
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import roc_auc_score,average_precision_score
import sys
sys.path.append('/home/minje/Projects/nlpfeatures')
os.chdir('/home/minje/Projects/nlpfeatures')
from features.word_features import ExtractWordFeatures
from features.tag_features import ExtractTags
from features.readability_features import ExtractReadability
from features.lexicon_features import ExtractVader,ExtractLIWC,ExtractEmpath,ExtractHateSonar,ExtractAllLinguistics,ExtractNRC
from features.lexicon_features import ExtractNGrams
from features.embedding_features import ExtractEmbeddingSimilarities
from sklearn.pipeline import FeatureUnion
from scipy.stats import pointbiserialr
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score,precision_score,f1_score,accuracy_score,average_precision_score
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from nltk.tokenize import TweetTokenizer
import os
from os.path import join
import re
from imblearn.over_sampling import RandomOverSampler
from preprocessing.customDataLoader import loadDataFromPandas
tokenize = TweetTokenizer().tokenize

def getScores(dim):
    base_dir = '/10TBdrive/minje/datasets/tinghy/'

    features = FeatureUnion([
        ('w1', ExtractWordFeatures()),
        ('r1', ExtractReadability()),
        ('l1', ExtractVader()),
        ('l3', ExtractAllLinguistics()),
        ('l4', ExtractLIWC()),
        ('l6', ExtractEmpath()),
        ('l10', ExtractHateSonar()),
        ('l12', ExtractNGrams(dimension=dim))
    ])

    with open('/home/minje/Projects/nlpfeatures/results/performance/xgboost/%s/all_0/model.pckl'%dim,'rb') as f:
        bst = pickle.load(f)

    df = pd.read_csv(join(base_dir,'tweets.csv'))
    messages = df['message'].values

    out = [features.transform([tokenize(text)]).squeeze() for text in messages]
    arr = np.stack(out)
    dtest = xgb.DMatrix(data=arr,feature_names=features.get_feature_names())
    scores = bst.predict(dtest)
    print(dim,len(scores))
    return scores


from multiprocessing import Pool

dims2 = ['Fun', 'Similarity', 'Respect', 'Knowledge transfer', 'Identity',
       'Trust', 'Social support', 'Power', 'Conflict', 'Romance']
dims = ['fun','similarity','respect','knowledge','identity','trust','social_support','power','conflict','romance']


try:
    pool = Pool(10)
    out = pool.map(getScores,dims)
finally:
    pool.close()

base_dir = '/10TBdrive/minje/datasets/tinghy/'
df = pd.read_csv(join(base_dir,'tweets.csv'))

for dim,scores in zip(dims,out):
    df[dim]=scores
df.to_csv(join(base_dir,'tweets_scores_xgboost.tsv'),sep='\t',index=False)
