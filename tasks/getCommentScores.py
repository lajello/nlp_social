# a code for calculating the comment scores of each comment in Reddit using LSTMs
# the output line is
# max score of comment, mean score of comment, user, subreddit name, region, location, sentence with max score

import os
from os.path import join
import sys
model_dir = '/home/minje/Projects/nlpfeatures'
sys.path.append(model_dir)
os.chdir(model_dir)

import torch
import gzip
import numpy as np
import ujson as json
from features.embedding_features import ExtractWordEmbeddings
from models.lstm import LSTMClassifier
from nltk.tokenize import TweetTokenizer
tokenize = TweetTokenizer().tokenize
from nltk import sent_tokenize
from preprocessing.customDataLoader import preprocessText,padBatch

def writeScores(tup):
    dim,n_file,n_set = tup

    # get score for comment
    from time import time
    start = time()
    cnt = 0
    data_dir = '/10TBdrive/minje/datasets/reddit/comments/geotagged/'
    save_dir = '/10TBdrive/minje/datasets/reddit/comments/scores/%s' % dim
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

        model_dir = '/10TBdrive/minje/models/lstm/'
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
                # elif ln>=(n_set+100000):
                elif ln >= (n_set + 1000):
                    break
                else:
                    try:
                        obj = json.loads(line.decode('utf-8'))
                        sents = sent_tokenize(preprocessText(obj['text']))
                        sents = [x for x in sents if len(x.split())>=5]
                        user, reg, loc, sub = obj['user'], obj['region'], obj['location'], obj['subreddit']
                    except:
                        continue

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
                        idx,mx,mean = np.argmax(scores),np.max(scores),np.mean(scores)
                        outf.write('\t'.join([str(round(mx, 3)),str(round(mean,3)), user, sub, reg, loc, sents[idx]]) + '\n')
                        cnt+=1
                    except:
                        continue
        print("Completed %s\tcount: %d\tminutes: %d"%(filename,cnt,int(time()/60-start/60)))

    return

if __name__=='__main__':
    from multiprocessing import Pool
    # writeScores(0)
    inputs = []
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
    dims = ['power']
    n_files = list(range(12))
    n_sets = list(range(0,12000000,100000))
    for n1 in n_files:
        for n2 in n_sets:
            for dim in dims:
                inputs.append((dim,n1,n2))
    try:
        pool = Pool(2)
        pool.map(writeScores,inputs)
    finally:
        pool.close()