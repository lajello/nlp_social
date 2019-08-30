
import sys
sys.path.append('/home/minje/Projects/nlpfeatures')
from features.word_features import ExtractWordFeatures
from features.readability_features import ExtractReadability
from features.lexicon_features import ExtractVader,ExtractLIWC,ExtractEmpath,ExtractHateSonar,ExtractAllLinguistics,ExtractNRC,ExtractNGrams
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
tokenize = TweetTokenizer().tokenize
import os
from os.path import join
import re
from preprocessing.customDataLoader import loadDataFromPandas

def cohen_d(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

def preprocessText(text):
    text = re.sub(pattern=r'\b[A-Za-z.\s]+:', string=text, repl='')  # remove characters - for round3
    text = re.sub(pattern=r'\<(.*?)\>', string=text, repl='')  # remove html tags - for round4
    text = re.sub(pattern=r'/?&gt;', string=text, repl='')  # remove html tags - for round4
    text = text.replace('"', '').strip()
    return text

def getBaseMAP(idx):
    from random import shuffle

    os.chdir('/home/minje/Projects/nlpfeatures')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d"%(idx%8)
    dims = ['conflict',
            'fun',
            'knowledge',
            'fun',
            'power',
            'respect',
            'romance',
            'similarity',
            'social_support',
            'trust']

    dim = dims[idx]
    print(idx)
    print(dim)

    tokenize = TweetTokenizer().tokenize
    input_ = 'h_text'
    data_dict = {}
    for dim in dims:
        X_tr, y_tr, X_v, y_v, X_t, y_t = loadDataFromPandas(dim=dim, balance=False)

        # calculate number of pos/neg samples for train/test
        pos = len([x for x in y_tr if x==1])
        neg = len(y_tr)-pos
        # print(dim,pos,neg)
        y_pred = y_tr
        shuffle(y_pred)
        y_pred = y_pred[:len(y_t)]
        map = average_precision_score(y_t,y_pred)
        print(dim,pos,neg,round(map,2))




    return

# produces trained results for all of the xgboost variations defined by 'mode'
def trainAllXGBVariations(idx):

    # hyperparameters
    is_cuda = True
    base_save_dir = '### where you want to save the models'
    base_save_dir = '/10TBdrive/minje/models/xgboost'

    # optional - set cuda device
    if is_cuda:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = "%d"%(idx%8)

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

    dim = dims[idx]
    print(idx)
    print(dim)


    # for mode, compile features in FeatureUnion object
    # for mode in ['all']:
    os.chdir('/home/minje/Projects/nlpfeatures')
    for mode in ['readability','vader','linguistics','empath','hate','ngrams','all']:
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

        # reset save_directory to consider both the dimension and mode
        save_dir = join(base_save_dir, dim, mode)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # load train/test/validation data
        X_tr, y_tr, X_v, y_v, X_t, y_t = loadDataFromPandas(dim=dim, balance=True)
        X_tr = features.transform(X_tr)
        X_t = features.transform(X_t)
        X_v = features.transform(X_v)

        feature_names = features.get_feature_names()

        # get cohen's d scores (optional)

        feat_corr = {}
        for i,name in enumerate(feature_names):
            pos_list = []
            neg_list = []
            for X_,y_ in [(X_tr,y_tr),(X_t,y_t),(X_v,y_v)]:
                for x_,label in zip(X_,y_):
                    s = x_[i]
                    if np.isnan(s):
                        continue
                    if label==1:
                        pos_list.append(s)
                    elif label==0:
                        neg_list.append(s)
            feat_corr[name] = cohen_d(pos_list,neg_list)

        # train model
        dtrain = xgb.DMatrix(data=X_tr,label=y_tr,feature_names=feature_names,nthread=10)
        deval = xgb.DMatrix(data=X_v,label=y_v,feature_names=feature_names,nthread=10)
        dtest = xgb.DMatrix(data=X_t,label=y_t,feature_names=feature_names,nthread=10)


        params = {'booster':'gbtree',
                  'nthread': 10,
                  'eta':0.003,
                  'max_depth':5,
                  'tree_method':'hist',
                  'objective':'binary:logistic',
                  'eval_metric':'logloss'}
        if is_cuda:
            params['tree_method'] = 'gpu_hist'

        bst = xgb.train(params=params,dtrain=dtrain,num_boost_round=2000,evals=[(deval,'eval')],
                        early_stopping_rounds=50,verbose_eval=1)
        y_prob = np.array(bst.predict(dtest))
        y_pred = np.array(y_prob>=0.5,dtype=int)

        y_true = y_t

        if np.sum(y_true) == 0:
            print("Skipping [%s], 0 positive samples" % dim)
            continue
        auc = round(roc_auc_score(y_true=y_true, y_score=y_prob), 3)
        rec = round(recall_score(y_true=y_true, y_pred=y_pred), 3)
        acc = round(accuracy_score(y_true=y_true, y_pred=y_pred), 3)
        pre = round(precision_score(y_true=y_true, y_pred=y_pred), 3)
        ap = round(average_precision_score(y_true=y_true,y_score=y_prob),3)
        print(dim,mode)
        print('AUC: ', round(auc, 2))
        print('REC: ', round(rec, 2))
        print('PRE: ', round(pre, 2))
        print('ACC: ', round(acc, 2))
        print('AP : ', round(ap, 2))
        #
        with open(join(save_dir, 'model.pckl'), 'wb') as f:
            pickle.dump(bst,f)
        #     # f.write('\t'.join([str(mode),str(auc), str(rec), str(pre), str(acc)]) + '\n')
        #
        #
        with open(join(save_dir, 'scores.tsv'), 'w') as f:
            f.write('\t'.join([str(mode),str(auc), str(rec), str(pre), str(acc), str(ap)]) + '\n')
        #
        importance = bst.get_score(importance_type='gain')
        importance = sorted([(k,v) for k,v in importance.items()])
        with open(join(save_dir,'feature_importance.tsv'),'w') as f:
            for k,v in importance:
                f.write('%s\t%1.2f\t%1.2f\n'%(k,v,feat_corr[k]))
    return



if __name__=='__main__':
    dims = ['Trust','Social support','Knowledge','Fun','Conflict',
                'Identity','Similarity','Power','Respect','Romance']

    for i in range(10):
        trainAllXGBVariations(i)
