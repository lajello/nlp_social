import os
from os.path import join
import re
from nltk import sent_tokenize
import pandas as pd
from time import time
import numpy as np
from collections import Counter
from sklearn.metrics import roc_auc_score,average_precision_score

dims = ['conflict','knowledge','trust','power','fun','romance','identity','similarity','respect','social_support']
print(len(dims))

import torch
import sys
file_dir = '/home/minje/Projects/nlpfeatures'
sys.path.append(file_dir)
os.chdir(file_dir)
from features.embedding_features import ExtractWordEmbeddings
from models.lstm import LSTMClassifier
from nltk.tokenize import TweetTokenizer
tokenize = TweetTokenizer().tokenize
from nltk import sent_tokenize
from preprocessing.customDataLoader import preprocessText

def getMovieScores(dim):
    # load pretrained model for dimension
    is_cuda = True
    model = LSTMClassifier(embedding_dim=300, hidden_dim=300)
    if is_cuda:
        model.cuda()
    model.eval()

    scores = {}
    print(dim)
    scores[dim] = []
    model_dir = ''
    for modelname in os.listdir(model_dir):
        if ('-best.lstm' in modelname) & (dim in modelname):
            best_state = torch.load(join(model_dir, modelname))
            model.load_state_dict(best_state)
            if 'glove' in modelname:
                em = ExtractWordEmbeddings('glove')
            elif 'word2vec' in modelname:
                em = ExtractWordEmbeddings('word2vec')
            elif 'fasttext' in modelname:
                em = ExtractWordEmbeddings('fasttext')
            print("Loaded model")
            break

    # load data
    with open('/10TBdrive/minje/datasets/movies/movie-lines.txt') as f:
        lines = f.readlines()
    lines = [x.strip().split(' +++$+++ ') for x in lines]
    df = pd.DataFrame(lines, columns=['line_id', 'char_id', 'movie_id', 'speaker', 'text']).dropna()
    start = time()
    cnt = 0
    out_mean = []
    out_max = []

    # get scores for each sample
    for text in df['text'].values:
        cnt+=1
        if cnt%1000==0:
            print(cnt,int(time()-start))
        sents = sent_tokenize(text)
        sent_scores = []
        for sent in sents:

            # change date into vectors
            input_ = em.obtain_vectors_from_sentence(tokenize(sent), True)
            input_ = torch.tensor(input_).float().unsqueeze(0)
            if is_cuda:
                input_ = input_.cuda()

            # get score between 0-1 for each sentence
            o = model(input_)
            o = torch.sigmoid(o).item()
            sent_scores.append(o)

        # get max and mean score of an email
        idx = np.argmax(sent_scores)
        max_sent = sents[idx]
        out_mean.append(np.mean(sent_scores))
        out_max.append(np.max(sent_scores))
    df['mean'] = out_mean
    df['max'] = out_max
    df.to_csv('/10TBdrive/minje/datasets/movies/scores/texts_%s.tsv'%dim, sep='\t',index=False)
    return



if __name__=='__main__':
    for dim in ['social_support','conflict','trust','knowledge','romance','identity','similarity',
                'power','respect','fun']:
        getMovieScores(dim)
