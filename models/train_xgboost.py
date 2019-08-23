
import sys
sys.path.append('/home/minje/Projects/nlpfeatures')
from features.word_features import ExtractWordFeatures
from features.readability_features import ExtractReadability
from features.lexicon_features import ExtractVader,ExtractLIWC,ExtractEmpath,ExtractHateSonar,ExtractAllLinguistics,ExtractNRC
from features.lexicon_features import ExtractNGrams
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

def trainXGBPerformance(idx):
    os.chdir('/home/minje/Projects/nlpfeatures')
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

    tokenize = TweetTokenizer().tokenize
    for n_round in range(10):
        # for mode in ['all']:
        for mode in ['liwc']:
        # for mode in ['word','readability','vader','linguistics','empath','hate','ngrams']:
            if mode=='word':
                features = FeatureUnion([
                    ('w1', ExtractWordFeatures()),
                ])
            elif mode=='readability':
                features = FeatureUnion([
                    ('r1', ExtractReadability()),
                ])
            elif mode=='vader':
                features = FeatureUnion([
                    ('l1', ExtractVader()),
                ])
            elif mode=='nrc':
                features = FeatureUnion([
                    ('l2', ExtractNRC()),
                ])
            elif mode=='linguistics':
                features = FeatureUnion([
                    ('l3', ExtractAllLinguistics()),
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
                    # ('l2', ExtractNRC()),
                    ('l3', ExtractAllLinguistics()),
                    ('l4', ExtractLIWC()),
                    ('l6', ExtractEmpath()),
                    ('l10', ExtractHateSonar()),
                    ('l12', ExtractNGrams(dimension=dim))
        ])

            # for dim in dims:
            input_ = 'h_text'

            data_dict = {}

            df = pd.read_csv('data/all-456.tsv', sep='\t')


            X_tr, y_tr, X_v, y_v, X_t, y_t = loadDataFromPandas(dim=dim, balance=True)
            X_tr = features.transform(X_tr)
            X_t = features.transform(X_t)
            X_v = features.transform(X_v)

            feature_names = features.get_feature_names()
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
                # pos_list = np.array(pos_list)
                # neg_list = np.array(neg_list)
                # pos_list = pos_list[np.logical_not(np.isnan(pos_list))]
                # neg_list = neg_list[np.logical_not(np.isnan(neg_list))]
                feat_corr[name] = cohen_d(pos_list,neg_list)

        # X = []
            #     y = []
            #     for X_,y_ in zip(df[input_].tolist(),df[dim].tolist()):
            #         if y_>=2:
            #             X.append(preprocessText(X_))
            #             y.append(1)
            #         elif y_==0:
            #             X.append(preprocessText(X_))
            #             y.append(0)
            #
            #     # X = X[:10]
            #     # print(len(X))
            #     X = features.transform(X)
            #     # sys.exit()
            #
            #     X_tr, X_t, y_tr, y_t = train_test_split(X,y,test_size=0.2)
            #     X_t, X_v, y_t, y_v = train_test_split(X_t,y_t, test_size=0.5)
            #
            #     # balance dataset
            #     ros = RandomOverSampler()
            #     X_tr,y_tr = ros.fit_resample(X_tr,y_tr)
            #     X_v, y_v = ros.fit_resample(X_v, y_v)
            #     X_t, y_t = ros.fit_resample(X_t, y_t)

            # for typ in ['train','test','valid']:
            #     df = pd.read_csv('data/%s.%s.tsv'%(dim,typ),sep='\t')
            #     # df = pd.read_csv('data/xgboost-format/%s.%s.tsv'%(dim,typ),sep='\t')
            #     X =
            #     X = features.transform(X) # feature transformation
            #     y = df['label']
            #     data_dict[typ] = (X,y)

            # X_tr,y_tr = data_dict['train']
            # X_t,y_t = data_dict['test']
            # X_v,y_v = data_dict['valid']
            # n_pos = np.array([x==1 for x in y_tr],dtype=int).sum()
            # n_neg = np.array([x==0 for x in y_tr],dtype=int).sum()
            # print('[%s] pos:%d/neg:%d'%(dim,n_pos,n_neg))

            # for n_round in range(10):
            save_dir = join(
                'results/performance/xgboost/%s/%s_%d' % (dim, mode, n_round))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # train model

            dtrain = xgb.DMatrix(data=X_tr,label=y_tr,feature_names=feature_names,nthread=10)
            deval = xgb.DMatrix(data=X_v,label=y_v,feature_names=feature_names,nthread=10)
            dtest = xgb.DMatrix(data=X_t,label=y_t,feature_names=feature_names,nthread=10)

            params = {'booster':'gbtree',
                      'nthread': 10,
                      'eta':0.003,
                      'max_depth':5,
                      'tree_method':'gpu_hist',
                      'objective':'binary:logistic',
                      'eval_metric':'logloss'}

            bst = xgb.train(params=params,dtrain=dtrain,num_boost_round=2000,evals=[(deval,'eval')],
                            early_stopping_rounds=50,verbose_eval=0)
            y_prob = np.array(bst.predict(dtest))
            y_pred = np.array(y_prob>=0.5,dtype=int)
            # y_prob = bst.

            # model = xgb.XGBClassifier(learning_rate=0.003, nthread=20, max_depth=5,
            #                           tree_method='gpu_hist', n_estimators=1000, verbosity=0)
            # model.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], eval_metric='auc', early_stopping_rounds=50, verbose=0)
            #
            # y_prob = model.predict_proba(X_t)[:, 1].tolist()
            # y_pred = model.predict(X_t).tolist()

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
                # for k,v in importance.items():
                    f.write('%s\t%1.2f\t%1.2f\n'%(k,v,feat_corr[k]))
            # imp = model.feature_importances_
            #
            # # save outputs
            # # names = features.get_feature_names()
            # with open(join(save_dir, 'feature-importance-%s.csv' % mode), 'w') as f:
            #     for i in range(len(feature_names)):
            #         name = feature_names[i]
            #         c_d = feat_corr[name]
            #         f.write(','.join([name,str(round(c_d,3)), str(round(imp[i], 4))]) + '\n')
    return

def trainXGB():
    os.chdir('/home/minje/Projects/nlpfeatures')
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
    for dim in dims:
    # for dim in dims:
        input_ = 'h_text'
        features = FeatureUnion([
            ('w1', ExtractWordFeatures()),
            # ('e1',ExtractEmbeddingSimilarities(emb_dim=dim,emb_type='glove')),
            # ('t1',ExtractTags()),
            ('r1', ExtractReadability()),
            ('l1', ExtractVader()),
            ('l2',ExtractNRC()),
            ('l3', ExtractAllLinguistics()),
            ('l4', ExtractLIWC()),
            ('l6', ExtractEmpath()),
            ('l10', ExtractHateSonar()),
            ('l12', ExtractNGrams())
        ])
        # features = ExtractNGrams(vocab_name='vocab-conflict.tsv')
        feature_names = features.get_feature_names()
        # print(len(feature_names),' features in total')

        data_dict = {}


        df = pd.read_csv('data/all-456.tsv',sep='\t')

        X_tr, y_tr, X_v, y_v, X_t, y_t = loadDataFromPandas(dim=dim,balance=True)
        X_tr = features.transform(X_tr)
        X_t = features.transform(X_t)
        X_v = features.transform(X_v)

    # X = []
    #     y = []
    #     for X_,y_ in zip(df[input_].tolist(),df[dim].tolist()):
    #         if y_>=2:
    #             X.append(preprocessText(X_))
    #             y.append(1)
    #         elif y_==0:
    #             X.append(preprocessText(X_))
    #             y.append(0)
    #
    #     # X = X[:10]
    #     # print(len(X))
    #     X = features.transform(X)
    #     # sys.exit()
    #
    #     X_tr, X_t, y_tr, y_t = train_test_split(X,y,test_size=0.2)
    #     X_t, X_v, y_t, y_v = train_test_split(X_t,y_t, test_size=0.5)
    #
    #     # balance dataset
    #     ros = RandomOverSampler()
    #     X_tr,y_tr = ros.fit_resample(X_tr,y_tr)
    #     X_v, y_v = ros.fit_resample(X_v, y_v)
    #     X_t, y_t = ros.fit_resample(X_t, y_t)



    # for typ in ['train','test','valid']:
        #     df = pd.read_csv('data/%s.%s.tsv'%(dim,typ),sep='\t')
        #     # df = pd.read_csv('data/xgboost-format/%s.%s.tsv'%(dim,typ),sep='\t')
        #     X =
        #     X = features.transform(X) # feature transformation
        #     y = df['label']
        #     data_dict[typ] = (X,y)

        # X_tr,y_tr = data_dict['train']
        # X_t,y_t = data_dict['test']
        # X_v,y_v = data_dict['valid']
        # n_pos = np.array([x==1 for x in y_tr],dtype=int).sum()
        # n_neg = np.array([x==0 for x in y_tr],dtype=int).sum()
        # print('[%s] pos:%d/neg:%d'%(dim,n_pos,n_neg))


        # train model
        model = xgb.XGBClassifier(learning_rate=0.003,nthread=20,max_depth=5,
                                  tree_method='gpu_hist',n_estimators=1000,verbosity=0)
        model.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], eval_metric='auc', early_stopping_rounds=50, verbose=0)

        y_prob = model.predict_proba(X_t)[:,1].tolist()
        y_pred = model.predict(X_t).tolist()

        y_true = y_t

        if np.sum(y_true)==0:
            print("Skipping [%s], 0 positive samples"%dim)
            continue
        rec = recall_score(y_true,y_pred)
        acc = accuracy_score(y_true,y_pred)
        pre = precision_score(y_true,y_pred)
        f1 = f1_score(y_true,y_pred)
        auc = roc_auc_score(y_true=y_true,y_score=y_prob)
        print("[%s] Test AUC: %1.3f, Recall: %1.3f, Precision: %1.3f, accuracy: %1.3f, F-1: %1.3f"%(dim,auc,rec,pre,acc,f1))

    return



def getScores(dim):
    dim = 'label'
    # dim = dim.lower().replace(' ','_')
    # load file
    import re
    from sklearn.metrics import roc_auc_score, recall_score, f1_score, precision_score

    # calculate for both round3 and round4
    # load round3
    df = pd.read_csv('data/conflict-45.tsv',sep='\t',header=None)
    df.columns = ['label','h_text']
    labels = ['label']
    # labels = [x for x in df.columns.tolist() if x.startswith('label:')&('other' not in x.lower())]
    y_dict = {label:[] for label in labels}
    for label in labels:
        y_dict[label].extend(df[label].tolist())

    X = []
    data_dim = 'h_text'
    for text in df[data_dim].values:
        text = re.sub(pattern=r'\b[A-Za-z.\s]+:',string=text,repl='') # remove characters - for round3
        text = re.sub(pattern=r'\<(.*?)\>',string=text,repl='') # remove html tags - for round4
        text = re.sub(pattern=r'/?&gt;',string=text,repl='') # remove html tags - for round4
        text = text.replace('"','').strip()
        X.append(text)

    # # load for round4
    # df = pd.read_csv('data/round4.tsv',sep='\t')
    # for text in df[data_dim].values:
    #     text = re.sub(pattern=r'\b[A-Za-z.\s]+:',string=text,repl='') # remove characters - for round3
    #     text = re.sub(pattern=r'\<(.*?)\>',string=text,repl='') # remove html tags - for round4
    #     text = re.sub(pattern=r'/?&gt;',string=text,repl='') # remove html tags - for round4
    #     text = text.replace('"','').strip()
    #     X.append(text)
    # for label in labels:
    #     y_dict[label].extend(np.array(df[label]>=2,dtype=int).tolist())


    X = [tokenize(text) for text in X]

    # print(len(X))
    # print(X[:5])
    # sys.exit()

    # import features
    features = FeatureUnion([
        ('w1',ExtractWordFeatures()),
        # ('t1',ExtractTags()),
        ('r1',ExtractReadability()),
        ('l1',ExtractVader()),
        ('l2',ExtractOffensive()),
        ('l3',ExtractMisspellings()),
        ('l4',ExtractLIWC()),
        ('l5',ExtractEmpathy()),
        ('l6',ExtractEmpath()),
        ('l7',ExtractHedging()),
        ('l8',ExtractIntegration()),
        ('l9',ExtractConcreteness()),
        ('l10',ExtractHateSonar()),
        ('l11',ExtractDifferentiation()),
        # ('l12',ExtractNGrams()),
        # ('l13',ExtractHashtags()),
        # ('l14',ExtractEmoji()),
        # ('em1',ExtractWordEmbeddings(emb_type='word2vec', emb_dir='/home/minje/embeddings', method='average')),
    ])
    feature_names = features.get_feature_names()


    X = features.transform(X) # feature transformation


    print("Transformed features")

    y = y_dict['label']
    # y = y_dict['label:%s'%dim]

    # y = np.array(df['label:%s'%dim]>=1,dtype=int).tolist() # I changed this to 1 to see how it differs when we get more labels

    # calculate the correlations for each feature
    corr_list = []
    y_bin = [x==1 for x in y]
    for i in range(len(feature_names)):
        r,pval = pointbiserialr(y_bin,X[:,i].tolist())
        corr_list.append((r,pval))


    y_pred = []
    y_prob = []
    y_true = []

    X = X.tolist()

    best_score = 0

    for i in range(1,2):
    # for i in range(10):
        print('training - ',i,dim)
        X_tr = X[0:i]+X[i:len(X)]
        y_tr = y[0:i]+y[i:len(y)]

        # to ensure that there is always at least 1 sample in the training and evaluation set
        y_v = [0]
        while (sum(y_tr)==0) | (sum(y_v)==0):
            X_tr,X_v,y_tr,y_v = train_test_split(X_tr,y_tr, test_size=0.1)

        X_t = X[i:i+1]
        y_t = y[i:i+1]

        # dtrain = xgb.DMatrix(X_tr,y_tr)
        # dtest = xgb.DMatrix(X_t,y_t)
        # deval = xgb.DMatrix(X_v,y_v)

        # X_tr, X_t, y_tr, y_t = train_test_split(X,y,test_size=0.1,random_state=np.random.randint(0,10000))

        n_pos = sum(y_tr)
        n_neg = len(y_tr)-n_pos
        print("Pos/neg rate: %d/%d"%(n_pos,n_neg))


        model = xgb.XGBClassifier(scale_pos_weight=n_neg/max(n_pos,1),learning_rate=0.03)

        print(len(X_tr),len(y_tr),len(X_t),len(y_t))
        # bst = xgb.train(params=param, dtrain=dtrain,
        #                 num_boost_round=num_round,
        #                 evals=[(deval,'eval')],
        #                 early_stopping_rounds=50)
        #
        # # test
        # ytrue = np.array(y_t, dtype=int)
        # yprob = np.array(bst.predict(dtest))

        X_tr = np.array(X_tr)
        y_tr = np.array(y_tr,dtype=int)
        model.fit(X_tr, y_tr, eval_set=[(X_v,y_v)], eval_metric='auc', early_stopping_rounds=50)

        if model.best_score>best_score:
            best_model = model
            best_score = model.best_score

        pred = model.predict(X_t)[0]
        prob = model.predict_proba(X_t)[0][1]

        # print(ytrue,yprob)
        # sys.exit()

        y_pred.append(pred)
        y_prob.append(prob)
        y_true.append(y_t[0])
        # y_pred.append(np.int(yprob[0]>=0.5))
        # y_prob.append(yprob[0])
        # y_true.append(ytrue[0])

    if sum(y_true)==0:
        y_true[0] = 1
    # y_true = y_true[:len(y_pred)]
    # print(y_pred,y_prob)

    # print("Dimension: ",dim)

    save_dir = join('results/round5/%s/xgboost/score'%data_dim)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("Saving scores for %s"%dim)
    with open(join(save_dir,'%s.csv'%dim),'w') as f:
        f.write('\t'.join(['y_true','y_pred','y_prob'])+'\n')
        for y1,y2,y3 in zip(y_true,y_pred,y_prob):
            f.write('\t'.join([str(x) for x in [y1,y2,y3]])+'\n')
        rec = recall_score(y_true=y_true, y_pred=y_pred)
        pre = precision_score(y_true=y_true, y_pred=y_pred)
        f1 = f1_score(y_true=y_true, y_pred=y_pred)
        auc = roc_auc_score(y_true=y_true, y_score=y_prob)
        f.write('AUC: %1.3f\n'%auc)
        f.write('F-1: %1.3f\n'%f1)
        f.write('REC: %1.3f\n' %rec)
        f.write('PRE: %1.3f\n' % pre)
    y_pos = np.sum(y)
    y_neg = len(y)-y_pos
    print("pos/neg ratio: 1/%1.2f"%(y_neg/y_pos))
    print("recall: %1.3f"%rec)
    print("prec  : %1.3f"%pre)
    print("F-1   : %1.3f"%f1)
    print("AUC   : %1.3f"%auc)

    # save feature importances
    imp = best_model.feature_importances_
    save_dir = join('results/round5/%s/xgboost/feature-importance'%data_dim)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("Saving feature importances for %s"%dim)

    # save outputs
    with open(join(save_dir,'%s.tsv'%dim),'w') as f:
        f.write('\t'.join(['feature-name','importance','corr','p-val'])+'\n')
        for i in range(len(imp)):
            r,pval = corr_list[i]
            line = [feature_names[i]] + [str(round(x,5)) for x in [imp[i],r,pval]]
            f.write('\t'.join(line)+'\n')
            # f.write(','.join([feature_names[i],str(round(imp[i],4)),str(round(r,2)),])+'\n')
    return

def getFeatureImportance(dim):
    # load file
    import re

    df = pd.read_csv('data/round4.tsv',sep='\t')
    tokenize = WordPunctTokenizer().tokenize

    X = []
    for text in df['text'].values:
        # text = re.sub(pattern=r'\b[A-Z]+:\s',string=text,repl='') # remove characters
        text = re.sub(pattern=r'\<(.*?)\>',string=text,repl='') # remove html tags - for round4
        X.append(tokenize(text))

    # import features
    features = FeatureUnion([
        ('w1',ExtractWordFeatures()),
        # ('t1',ExtractTags()),
        ('r1',ExtractReadability()),
        ('l1',ExtractVader()),
        ('l2',ExtractOffensive()),
        ('l3',ExtractMisspellings()),
        ('l4',ExtractLIWC()),
        ('l5',ExtractEmpathy()),
        ('l6',ExtractEmpath()),
        ('l7',ExtractHedging()),
        ('l8',ExtractIntegration()),
        ('l9',ExtractConcreteness()),
        ('l10',ExtractHateSonar()),
        ('l11',ExtractDifferentiation()),
        # ('l12',ExtractNGrams()),
        # ('l13',ExtractHashtags()),
        # ('l14',ExtractEmoji()),
        # ('em1',ExtractWordEmbeddings(emb_type='word2vec', emb_dir='/home/minje/embeddings', method='average')),
    ])

    names = features.get_feature_names()

    X = features.transform(X) # feature transformation
    X = X.tolist()
    print("Transformed features")

    # y = df['label:%s'%dim].values.tolist()
    y = df['label']
    y_pred = []
    y_prob = []
    y_true = []
    # X = X[:10]
    # y = y[:10]

    X_tr, X_v, y_tr, y_v = train_test_split(X, y, test_size=0.1)

    # X_tr = X
    # y_tr = y
    model = xgb.XGBClassifier()
    X_tr = np.array(X_tr)
    y_tr = np.array(y_tr, dtype=int)
    model.fit(X_tr, y_tr, eval_set=[(X_v,y_v)], eval_metric='auc', early_stopping_rounds=50)

    imp = model.feature_importances_
    # print(imp)
    # print(type(imp))
    # print(imp.shape)
    # print(names,len(names))

    # save outputs
    save_dir = '/home/minje/Projects/nlpfeatures/results/whole-sentences/xgboost/feature-importance'
    save_dir = join('results/round4/', dim, '/xgboost/feature-importance')
    with open(join(save_dir,'%s.csv'%dim),'w') as f:
        for i in range(len(imp)):
            f.write(','.join([names[i],str(round(imp[i],4))])+'\n')

    # for i in range(len(y)):
    #     X_tr = X[0:i]+X[i+1:len(X)]
    #     X_t = X[i:i+1]
    #     y_tr = y[0:i]+y[i+1:len(y)]
    #     y_t = y[i:i+1]
    #     y_true.extend(y_t)
    #     # X_tr, X_t, y_tr, y_t = train_test_split(X,y,test_size=0.1,random_state=np.random.randint(0,10000))
    #
    #     pred = model.predict(X_t)[0]
    #     prob = model.predict_proba(X_t)[0][1]
    #     y_pred.append(pred)
    #     y_prob.append(prob)
    # # print(y_pred,y_prob)
    #
    # print("Dimension: ",dim)
    # with open('results/whole-sentences/xgboost/%s.csv'%dim,'w') as f:
    #     f.write('\t'.join(['y_true','y_pred','y_prob'])+'\n')
    #     for y1,y2,y3 in zip(y_true,y_pred,y_prob):
    #         f.write('\t'.join([str(x) for x in [y1,y2,y3]])+'\n')
    #     f.write('AUC: %1.3f\n'%roc_auc_score(y_true=y_true,y_score=y_prob))
    #     f.write('REC: %1.3f\n' % recall_score(y_true=y_true,y_pred=y_pred))
    # y_pos = np.sum(y)
    # y_neg = len(y)-y_pos
    # print("pos/neg ratio: 1/%1.2f"%(y_neg/y_pos))
    # print("Average recall: %1.3f"%np.mean(recall_list))
    # print("Average AUC   : %1.3f"%np.mean(auc_list))
    return




if __name__=='__main__':
    dims = ['Trust','Social support','Knowledge','Fun','Conflict',
                'Identity','Similarity','Power','Respect','Romance']

    trainXGBPerformance(0) # the input is idx, only is the index for which dimension to consider