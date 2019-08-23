import os
from os.path import join
import pandas as pd
import sys
sys.path.append(os.getcwd())
from models.lstm import LSTMClassifier
from features.embedding_features import ExtractWordEmbeddings
import torch
from nltk.tokenize import TweetTokenizer
from nltk import sent_tokenize
import numpy as np
tokenize = TweetTokenizer().tokenize
from time import time

def getEnronScores(dim):
    is_cuda = True
    model = LSTMClassifier(embedding_dim=300, hidden_dim=300)
    if is_cuda:
        model.cuda()
    model.eval()

    scores = {}
    print(dim)
    scores[dim] = []
    model_dir = 'results/performance/lstm/'
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

    df = pd.read_csv('/10TBdrive/minje/datasets/enron-texts/all-date-texts.tsv', sep='\t').dropna()
    out = []
    start = time()
    cnt = 0
    for day,text in df[['date','text']].values:
        cnt+=1
        if cnt%1000==0:
            print(cnt,int(time()-start))
        # if cnt>10000:
        #     break
        sents = sent_tokenize(text)
        sent_scores = []
        for sent in sents:
            input_ = em.obtain_vectors_from_sentence(tokenize(sent), True)
            input_ = torch.tensor(input_).float().unsqueeze(0)
            if is_cuda:
                input_ = input_.cuda()
            o = model(input_)
            o = torch.sigmoid(o).item()
            sent_scores.append(o)
        idx = np.argmax(sent_scores)
        max_sent = sents[idx]
        out.append((day,dim,max_sent,np.mean(sent_scores),np.max(sent_scores)))
    df_out = pd.DataFrame(out,columns=['date','dimension','max_sent','mean','max'])
    df_out.to_csv('/10TBdrive/minje/datasets/enron-texts/all-date-texts_%s.tsv'%dim, sep='\t',index=False)
    return



if __name__=='__main__':
    getEnronScores(sys.argv[1])