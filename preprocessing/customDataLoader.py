import numpy as np
import re
import pandas as pd
import os
from nltk.tokenize import TweetTokenizer
from sklearn.utils import shuffle
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
tokenize = TweetTokenizer().tokenize


def preprocessText(text):
    text = re.sub(pattern=r'\b[A-Za-z.\s]+:', string=text, repl='')  # remove characters - for round3
    text = re.sub(pattern=r'\<(.*?)\>', string=text, repl='')  # remove html tags - for round4
    text = re.sub(pattern=r'/?&gt;', string=text, repl='')  # remove html tags - for round4
    text = re.sub(pattern=r'(\n|\t)',string=text, repl=' ') # remove cases of line splitters
    text = text.replace('"', '').strip()
    return text


def loadDataFromPandas(field='h_text', dim='conflict', balance=False, window=0):
    os.chdir('/home/minje/Projects/nlpfeatures')
    df = pd.read_csv('/home/minje/Projects/nlpfeatures/data/all-456.tsv', sep='\t')
    #     df['round']=[int(x) for x in df['round']]
    # df = df[df['round']!=6]
    dim_col_idx = df.columns.tolist().index(dim)
    text_col_idx = df.columns.tolist().index(field)
    X = []
    y = []

    # append positive/negative samples to X and y lists
    for line in df.values:
        label = line[dim_col_idx]
        if (label>=2)|(label==0):
            text = tokenize(preprocessText(line[text_col_idx]))
            if (window>0)&(len(text)>window):
                for i in range(window,len(text)):
                    X.append(text[i-window:i])
                    y.append(int(label>=2))
            else:
                X.append(text)
                y.append(int(label>=2))

    X_tr, X_t, y_tr, y_t = train_test_split(X, y, test_size=0.2, random_state=42)
    X_t, X_v, y_t, y_v = train_test_split(X_t, y_t, test_size=0.5, random_state=42)

    # balance train/test/dev data if 'balance' flag is up
    if balance:
        # X_tr
        y_pos_idx = [i for i, v in enumerate(y_tr) if v == 1]
        y_neg_idx = [i for i, v in enumerate(y_tr) if v == 0]
        y_pos_idx = np.random.choice(y_pos_idx, len(y_neg_idx), replace=True).tolist()
        X_tr = [X_tr[i] for i in y_pos_idx + y_neg_idx]
        y_tr = [1] * len(y_pos_idx) + [0] * len(y_neg_idx)

        # X_t
        # y_pos_idx = [i for i, v in enumerate(y_t) if v == 1]
        # y_neg_idx = [i for i, v in enumerate(y_t) if v == 0]
        # y_pos_idx = np.random.choice(y_pos_idx, len(y_neg_idx), replace=True).tolist()
        # X_t = [X_t[i] for i in y_pos_idx + y_neg_idx]
        # y_t = [1] * len(y_pos_idx) + [0] * len(y_neg_idx)

        # X_v
        y_pos_idx = [i for i, v in enumerate(y_v) if v == 1]
        y_neg_idx = [i for i, v in enumerate(y_v) if v == 0]
        y_pos_idx = np.random.choice(y_pos_idx, len(y_neg_idx), replace=True).tolist()
        X_v = [X_v[i] for i in y_pos_idx + y_neg_idx]
        y_v = [1] * len(y_pos_idx) + [0] * len(y_neg_idx)

    return X_tr, y_tr, X_v, y_v, X_t, y_t


def padBatch(list_of_list_of_arrays,max_seq=None):
    """
    A function that returns a numpy array that creates a B @ (seq x dim) sized-tensor
    :param list_of_list_of_arrays:
    :return:
    """

    if max_seq:
        list_of_list_of_arrays = [X[:max_seq] for X in list_of_list_of_arrays]


    # get max length
    mx = max(len(V) for V in list_of_list_of_arrays)

    # get array dimension by looking at the 1st sample
    array_dimensions = [len(V[0]) for V in list_of_list_of_arrays]
    assert min(array_dimensions)==max(array_dimensions), "Dimension sizes do not match within the samples!"
    dim_size = array_dimensions[0]

    # get empty array to put in
    dummy_arr = [0]*dim_size

    # create additional output
    out = []
    for V in list_of_list_of_arrays:
        V = V.tolist()
        out.append(V+[dummy_arr]*(mx-len(V)))

    # return
    return np.array(out)

def getLeaveOneOutData(X,y,i=0,test_size=1):
    """
    Returns the train/test set that leaves only one out
    :param X:
    :param y:
    :param i:
    :return:n
    """
    X_tr = X[0:i*test_size] + X[(i+1)*test_size:len(X)]
    X_t = X[i*test_size:(i+1)*test_size]
    y_tr = y[0:i*test_size] + y[(i+1)*test_size:len(y)]
    y_t = y[i*test_size:(i+1)*test_size]

    X_tr, X_val, y_tr, y_val = train_test_split(X_tr,y_tr, test_size=0.1)

    return X_tr, y_tr, X_val, y_val, X_t, y_t

def batchify(X, y, batch_size=50):
    """
    Given a list of data samples, returns them in a batch
    :param X:
    :param y:
    :param batch_size:
    :return:
    """
    out = []
    n_batches = int(np.ceil(len(X)/batch_size))
    for i in range(n_batches):
        X_b = X[i*batch_size:(i+1)*batch_size]
        y_b = y[i*batch_size:(i+1)*batch_size]
        out.append((X_b,y_b))

    return out

