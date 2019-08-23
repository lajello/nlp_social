import os
from os.path import join
import sys
from time import time
import pandas as pd

def train(dim):
    # sys.path.append('/home/minje/Projects/nlpfeatures')

    import torch
    from torch import nn,optim
    from features.embedding_features import ExtractWordEmbeddings
    from preprocessing.customDataLoader import batchify,padBatch,loadDataFromPandas
    from models.lstm import LSTMClassifier
    from sklearn.metrics import roc_auc_score,recall_score,accuracy_score,precision_score,average_precision_score
    from sklearn.utils import shuffle

    hyperparameter_embedding = ['word2vec','glove','fasttext']
    hyperparameter_lr = [0.01,0.001,0.0001,0.00001]

    is_cuda = True

    batch_size = 50
    start = time()
    field = 'h_text' # or 'text

    # load training-test-validation data
    X_tr, y_tr, X_val, y_val, X_t, y_t = loadDataFromPandas(field=field, dim=dim, balance=True)
    print("Train: %d/%d, Test: %d/%d, Valid: %d/%d" % (len([x for x in y_tr if x == 1]),
                                                       len([x for x in y_tr if x == 0]),
                                                       len([x for x in y_t if x == 1]),
                                                       len([x for x in y_t if x == 0]),
                                                       len([x for x in y_val if x == 1]),
                                                       len([x for x in y_val if x == 0])))

    if dim in {'conflict','identity','respect'}:
        emb_type='glove'
    elif dim in {'fun','knowledge','power','romance','similarity',
                 'social_support'}:
        emb_type='word2vec'
    elif dim in {'trust'}:
        emb_type='fasttext'

    if dim in {'conflict','respect','social_support'}:
        lr = 0.01
    elif dim in {'fun','romance'}:
        lr = 0.001
    elif dim in {'knowledge','power','trust'}:
        lr = 0.0001
    elif dim in {'identity','similarity'}:
        lr = 0.00001

    # save_dir = join('/home/minje/Projects/nlpfeatures/results',dim)
    save_dir = 'results/performance/lstm'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(join(save_dir, '%s-%s-scores.tsv'%(dim,emb_type)), 'w') as f:
        f.write('\t'.join(['embedding', 'lr', 'AUC', 'recall','precision', 'ACC','AUCPR']) + '\n')


    em = ExtractWordEmbeddings(emb_type)
    best_score = 0.0
    for n2 in range(1):
        for n in range(10):
    #         model = BiLSTMClassifier(embedding_dim=300, hidden_dim=300)
            model = LSTMClassifier(embedding_dim=300, hidden_dim=300)
            if is_cuda:
                model.cuda()
            optimizer = optim.Adam(model.parameters(),lr=lr)
            flag = True
            old_val = 10000 # previous validation error
            epoch = 0
            cnt_decrease = 0 # reset to 0 whenever the lowest validation error changes
            while(flag):
                tr_loss = 0.0
                epoch+=1
                if (epoch > 100) | (cnt_decrease > 5):
                    break
                # train
                model.train()
                X_tr, y_tr = shuffle(X_tr, y_tr)
                tr_batches = batchify(X_tr, y_tr, batch_size)
                for X_b,y_b in tr_batches:
                    loss_fn = nn.BCEWithLogitsLoss()
                    inputs = torch.tensor(padBatch([em.obtain_vectors_from_sentence(sent,True) for sent in X_b])).float()
                    targets = torch.tensor(y_b,dtype=torch.float32)
                    if is_cuda:
                        inputs,targets = inputs.cuda(),targets.cuda()
                    outputs = model(inputs)
                    loss = loss_fn(outputs,targets) # error here
                    optimizer.zero_grad()
                    loss.backward()
                    tr_loss += loss.item()
                    optimizer.step()

                # validate
                model.eval()
                inputs = torch.tensor(padBatch([em.obtain_vectors_from_sentence(sent, True) for sent in X_val])).float()
                targets = torch.tensor(y_val, dtype=torch.float32)
                if is_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                    # inputs_inv = inputs_inv.cuda()
                outputs = model(inputs)
                # outputs = model(inputs, inputs_inv)
                loss = loss_fn(outputs,targets).item()
                print("%s-%s-%s Epoch %d: %1.3f"%(dim,emb_type,str(lr),epoch,loss))
                cnt_decrease += 1
                if loss<old_val:
                    # save this model
                    best_state = model.state_dict()
                    torch.save(best_state, join(save_dir,'%s-%s.lstm.pth') %(dim,emb_type))
                    old_val = loss
                    # print(model.state_dict()['lstm.weight_ih_l0'][0][:5])
                    print("Updated model")
                    cnt_decrease = 0

            # evaluate using best model
            best_state = torch.load(join(save_dir,'%s-%s.lstm.pth') %(dim,emb_type))
            model.load_state_dict(best_state)
            model.eval()
            inputs = torch.tensor(padBatch([em.obtain_vectors_from_sentence(sent, True) for sent in X_t])).float()
            if is_cuda:
                inputs = inputs.cuda()
            scores = torch.sigmoid(model(inputs))
            y_true = y_t
            y_prob = scores.tolist()
            y_pred = (scores >= 0.5).tolist()
            auc = round(roc_auc_score(y_true=y_true, y_score=y_prob),3)
            rec = round(recall_score(y_true=y_true, y_pred=y_pred),3)
            acc = round(accuracy_score(y_true=y_true, y_pred=y_pred),3)
            pre = round(precision_score(y_true=y_true, y_pred=y_pred),3)
            ap = round(average_precision_score(y_true=y_true, y_score=y_prob),3)
            print('%s-%s-%s'%(dim,emb_type,str(lr)))
            print(n2)
            print('AUC: ', round(auc, 2))
            print('REC: ', round(rec, 2))
            print('PRE: ', round(pre, 2))
            print('ACC: ', round(acc, 2))
            print('AP : ', round(ap, 2))

            with open(join(save_dir, '%s-%s-scores.tsv'%(dim,emb_type)), 'a') as f:
                f.write('\t'.join([emb_type,str(lr),str(auc),str(rec),str(pre),str(acc),str(ap)])+'\n')
            if ap>best_score:
                torch.save(best_state, join(save_dir, '%s-%s-best.lstm.pth') % (dim, emb_type))
                best_score = ap
    return

if __name__=='__main__':
    import sys

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

    train(sys.argv[1]) # train by typing dimension name shown in 'dims'

