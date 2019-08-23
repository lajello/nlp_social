import os
from os.path import join
import sys
sys.path.append(os.getcwd())

import torch
import gzip
import gc
import ujson as json
from features.embedding_features import ExtractWordEmbeddings
from models.lstm import LSTMClassifier
from nltk.tokenize import TweetTokenizer
tokenize = TweetTokenizer().tokenize
from nltk import sent_tokenize
from preprocessing.customDataLoader import preprocessText,padBatch

# from torch import nn, optim
# from preprocessing.customDataLoader import getLeaveOneOutData,batchify,padBatch
# import numpy as np
# from sklearn.metrics import roc_auc_score,recall_score,accuracy_score
# from sklearn.utils import shuffle
# from time import time
# import pandas as pd
# from collections import Counter
# import re
# from sklearn.model_selection import train_test_split
# from imblearn.under_sampling import RandomUnderSampler
# from preprocessing.customDataLoader import loadDataFromPandas,preprocessText

def writeScores(tup):
    dim,n_file,n_set = tup
    # dim = 'conflict'

    # get score for comment
    from time import time
    start = time()
    cnt = 0
    data_dir = '/10TBdrive/minje/datasets/reddit/comments/processed/'
    save_dir = '/10TBdrive/minje/datasets/reddit/comments/expression-filtering/%s' % dim
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, filename in enumerate(sorted(os.listdir(data_dir))):
        if i != n_file:
            continue
        save_filename = join(save_dir, '%s-%s-%d-%d.tsv' % (dim, filename.split('.')[0],
                                                            n_set, n_set + 100000))
        if os.path.exists(save_filename):
            print("%s exists! Skipping..."%save_filename)
            return

        is_cuda = True
        model = LSTMClassifier(embedding_dim=300, hidden_dim=300)
        if is_cuda:
            model.cuda()
        model.eval()

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

        with gzip.open(join(data_dir, filename)) as f, \
                open(save_filename, 'w') as outf:
            print(save_filename)
            for ln, line in enumerate(f):
                if ln<n_set:
                    continue
                elif ln>=(n_set+100000):
                    break
                else:
                # if (ln % 10000 == 0) & (ln >0):
                #     print(dim, filename, ln, int((time() - start)/60), cnt)
                #     gc.collect()
                    try:
                        obj = json.loads(line.decode('utf-8'))
                        sents = sent_tokenize(preprocessText(obj['text']))
                        user, reg, loc = obj['user'], obj['region'], obj['location']
                    except:
                        continue
                    # inputs = torch.tensor(padBatch([em.obtain_vectors_from_sentence(sent, True) for sent in sents])).float()
                    # if is_cuda:
                    #     inputs = inputs.cuda()
                    # scores = torch.sigmoid(model(inputs)).tolist()
                    # if type(scores)==float:
                    #     s = scores
                    # else:
                    #     s = max(scores)
                    # if s>=0.9:
                    #     cnt += 1
                    #     outf.write('\t'.join([str(round(s, 3)), user, reg, loc, sents]) + '\n')
                    #
                    #
                    scores = []
                    for sent in sents:
                        # sent = preprocessText(sent)
                        input_ = em.obtain_vectors_from_sentence(sent.split(), True)
                        input_ = torch.tensor(input_).float().unsqueeze(0)
                        if is_cuda:
                            input_ = input_.cuda()
                        s = model(input_)
                        s = torch.sigmoid(s).item()
                        scores.append(s)
                    try:
                        s = max(scores)
                        if s >= 0.6:
                            cnt += 1
                            outf.write('\t'.join([str(round(s, 3)), user, reg, loc, ' '.join(sents)]) + '\n')
                    except:
                        continue
        print("Completed %s\tcount: %d\tminutes: %d"%(filename,cnt,int(time()/60-start/60)))

    return

if __name__=='__main__':
    from multiprocessing import Pool
    # writeScores(0)
    inputs = []
    # dims = ['social_support',
    #         'conflict',
    #         'trust',
    #         'fun',
    #         'similarity',
    #         'identity',
    #         'respect',
    #         'romance',
    #         'knowledge',
    #         'power']
    dims = ['identity']
    n_files = list(range(12))
    n_sets = list(range(0,12000000,100000))
    for n1 in n_files:
        for n2 in n_sets:
            for dim in dims:
                inputs.append((dim,n1,n2))
    try:
        pool = Pool(4)
        pool.map(writeScores,inputs)
    finally:
        pool.close()