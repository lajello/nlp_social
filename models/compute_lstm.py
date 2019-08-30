# a code for running LSTMs to get scores for any text data

import os
from os.path import join
import torch
import sys


code_dir = '### where the models and features codes are stored'
code_dir = '/home/minje/Projects/nlpfeatures'
sys.path.append(code_dir)
os.chdir(code_dir)
from features.embedding_features import ExtractWordEmbeddings
from models.lstm import LSTMClassifier
from nltk.tokenize import TweetTokenizer
tokenize = TweetTokenizer().tokenize


dims = ['conflict','knowledge','trust','power','fun','romance','identity','similarity','respect','social_support']
print(dims)

# hyperparameter
dim = 'conflict' # select dimension from one of the dims above

# load pretrained model for dimension
is_cuda = True
model = LSTMClassifier(embedding_dim=300, hidden_dim=300)
if is_cuda:
    model.cuda()
model.eval()

model_dir = '### where the saved models are stored'
model_dir = '/10TBdrive/minje/models/lstm'

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


# replace this with any text data that you have
sentences = ['this is a sentence',
             'i hate what you do',
             'this is a good idea',
             'that is a bad idea',
             'i love that idea']


# get scores for each sample
scores = []
for sent in sentences:

    # change data into vectors
    input_ = em.obtain_vectors_from_sentence(tokenize(sent), True)
    input_ = torch.tensor(input_).float().unsqueeze(0)
    if is_cuda:
        input_ = input_.cuda()

    # get score between 0-1 for each sentence
    o = model(input_)
    o = torch.sigmoid(o).item()
    scores.append(o)

for s,sent in zip(scores,sentences):
    print(round(s,3),sent)
