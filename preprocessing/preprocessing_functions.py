import os
from os.path import join
import re
from time import time

def preprocessText(text):
    text = re.sub(pattern=r'\b[A-Za-z.\s]+:', string=text, repl=' ')  # remove characters - for round3
    text = re.sub(pattern=r'\<(.*?)\>', string=text, repl=' ')  # remove html tags - for round4
    text = re.sub(pattern=r'/?&gt;', string=text, repl=' ')  # remove html tags - for round4
    text = text.replace('"', '').strip()
    return text

def word2vec4gensim(file_dir):
    """
    A function that modifies the pretrained word2vec or fasttext file so it could be integrated with this framework

    [Note] You can download the vectors used in this code at the following directories
    word2vec: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
    fasttext: https://fasttext.cc/docs/en/english-vectors.html or
    https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz (make sure to unzip the files)

    :param file_dir: file directory of the downloaded word2vec.bin file
    e.g., file_dir='/home/USERNAME/embeddings/word2vec/GoogleNews-vectors-negative300.bin'
    :return: None
    """

    from gensim.models import KeyedVectors

    # load the vectors on gensim
    assert file_dir.endswith('.bin')|file_dir.endswith('.vec'), "Input file should be either a .bin or .vec"
    model = KeyedVectors.load_word2vec_format(file_dir,binary=file_dir.endswith('.bin'))
    # save only the .wv part of the model, it's much faster
    new_file_dir = file_dir.replace('.bin','.wv')
    model.wv.save(new_file_dir)
    # delete the original .bin file
    os.remove(file_dir)
    print("Removed previous file ",file_dir)

    # try loading the new file
    model = KeyedVectors.load(new_file_dir, mmap='r')
    print("Loaded in gensim! %d word embeddings, %d dimensions"%(len(model.vocab),len(model['a'])))
    return

def glove4gensim(file_dir):
    """
    A function that modifies the pretrained GloVe file so it could be integrated with this framework

    [Note] You can download the vectors used in this code at
    https://nlp.stanford.edu/projects/glove/ (make sure to unzip the files)

    :param file_dir: file directory of the downloaded file
    e.g., file_dir='/home/USERNAME/embeddings/word2vec/GoogleNews-vectors-negative300.bin'
    :return: None
    """

    from gensim.models import KeyedVectors
    from gensim.scripts.glove2word2vec import glove2word2vec

    # load the vectors on gensim
    assert file_dir.endswith('.txt'), "For downloaded GloVe, the input file should be a .txt"
    glove2word2vec(file_dir,file_dir.replace('.txt','.vec'))
    file_dir = file_dir.replace('.txt','.vec')
    model = KeyedVectors.load_word2vec_format(file_dir,binary=file_dir.endswith('.bin'))
    # save only the .wv part of the model, it's much faster
    new_file_dir = file_dir.replace('.bin','.wv')
    model.wv.save(new_file_dir)
    # delete the original .bin file
    os.remove(file_dir)
    print("Removed previous file ",file_dir)

    # try loading the new file
    model = KeyedVectors.load(new_file_dir, mmap='r')
    print("Loaded in gensim! %d word embeddings, %d dimensions"%(len(model.vocab),len(model['a'])))
    return

def train_ngrams(file_dir,save_dir='lexicons/',vocab_size=10000,max_n=4, min_count=5, threshold=5):
    os.chdir('/home/minje/Projects/nlpfeatures')
    """
    A function that creates the files needed for running n-grams
    :param file_dir: the directory of the file to create the n-gram from
    :param save_dir: the directory to save
    :param vocab_size: The maximum size of the vocabulary (default=10000)
    :param max_n: The size of which n-grams to look at (default=3: n=1,2,3 grams will be considered)
    :param min_count: How many times should the n-gram appear
    :param threshold: Computed threshold (the higher it is, the fewer n-grams you get)
    :return:
    """
    import pickle
    from gensim.models.phrases import Phraser,Phrases
    from nltk.tokenize import TweetTokenizer
    import pandas as pd
    from nltk import sent_tokenize
    tokenize = TweetTokenizer().tokenize

    df = pd.read_csv(file_dir,sep='\t')
    sentences = []
    for sents in df['text'].values:
        sents = sent_tokenize(preprocessText(sents.lower()))
        for sent in sents:
            sentences.append(tokenize(sent))
    # sentences = [tokenize(preprocessText(x.lower())) for x in df['text']]
    words = []
    for sent in sentences:
        words.extend(sent)
    print(words[:100])

    # with open(file_dir) as f:
    #     words = tokenize(f.read().lower())
    print("Tokenized all words! %d tokens"%len(words))
    print("%d unique phrases" % len(set(words)))

    # train bigrams
    bigrams = Phrases(sentences, min_count=min_count, threshold=threshold)
    phraser = Phraser(bigrams)
    with open(join(save_dir, 'phraser-bi.pckl'), 'wb') as f:
        pickle.dump(phraser, f)
    words = phraser[words]
    sentences = phraser[sentences]
    print("Tokenized all words! %d tokens"%len(words))
    print("%d unique phrases" % len(set(words)))
    print(words[:100])

    # train trigrams
    trigrams = Phrases(sentences,min_count=min_count, threshold=threshold)
    phraser = Phraser(trigrams)
    with open(join(save_dir, 'phraser-tri.pckl'), 'wb') as f:
        pickle.dump(phraser, f)
    words = phraser[words]
    print("Tokenized all words! %d tokens"%len(words))
    print("%d unique phrases" % len(set(words)))
    print(words[:100])

    # get vocab
    from collections import Counter
    cn = Counter(words)
    with open(join(save_dir,'ngram-vocab.txt'), 'w') as f:
        for k,v in cn.most_common(vocab_size):
            f.write(k+'\n')
    return

def saveLogOddNGrams():
    # first get all text that are positive and negative
    import sys
    sys.path.append('/home/minje/Projects/nlpfeatures')
    os.chdir('/home/minje/Projects/nlpfeatures')
    from features.lexicon_features import ExtractNGrams
    ngrams = ExtractNGrams()
    from nltk.tokenize import TweetTokenizer
    from collections import Counter
    import pandas as pd
    tokenize = TweetTokenizer().tokenize

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
    for dim in dims:
        df = pd.read_csv('data/all-456.tsv',sep='\t')
        df['text'] = [ngrams.tri_phraser[ngrams.bi_phraser[tokenize(preprocessText(x.lower()))]] for x in df['h_text']]
        pos_words = []
        neg_words = []
        for words in df[df[dim]>=2]['text'].values:
            pos_words.extend(words)
        for words in df[df[dim]==0]['text'].values:
            neg_words.extend(words)
        cn_pos = Counter(pos_words)
        cn_neg = Counter(neg_words)
        returnLogOdds(cn_pos,cn_neg,dim)

    return

def returnLogOdds(cn_pos,cn_neg,dim):
    import operator
    from collections import defaultdict
    import math
    prior = cn_pos + cn_neg
    counts1 = cn_pos
    counts2 = cn_neg
    sum_pos = sum([v for v in cn_pos.values()])
    sum_neg = sum([v for v in cn_neg.values()])

    sigmasquared = defaultdict(float)
    sigma = defaultdict(float)
    delta = defaultdict(float)

    for word in prior.keys():
        prior[word] = int(prior[word] + 0.5)

    for word in counts2.keys():
        counts1[word] = int(counts1[word] + 0.5)
        if prior[word] == 0:
            prior[word] = 1

    for word in counts1.keys():
        counts2[word] = int(counts2[word] + 0.5)
        if prior[word] == 0:
            prior[word] = 1

    n1 = sum(counts1.values())
    n2 = sum(counts2.values())
    nprior = sum(prior.values())

    for word in prior.keys():
        # if prior[word] == 0 and (counts2[word] > 10):
        # prior[word] = 1
        if prior[word] > 0:
            l1 = float(counts1[word] + prior[word]) / ((n1 + nprior) - (counts1[word] + prior[word]))
            l2 = float(counts2[word] + prior[word]) / ((n2 + nprior) - (counts2[word] + prior[word]))
            sigmasquared[word] = 1 / (float(counts1[word]) + float(prior[word])) + 1 / (
                        float(counts2[word]) + float(prior[word]))
            sigma[word] = math.sqrt(sigmasquared[word])
            delta[word] = (math.log(l1) - math.log(l2)) / sigma[word]
    beqout = delta

    # print(beqout)

    save_dir = '/home/minje/Projects/nlpfeatures/lexicons/vocab-%s.tsv'%dim
    outf = open(save_dir, 'w')
    for key, val in sorted(beqout.items(), key=operator.itemgetter(1), reverse=True):
        v1 = cn_pos[key]
        v2 = cn_neg[key]
        if (v2 / sum_neg) < 0.001:
            continue
        else:
            val1 = '%d/%d'%(v1,sum_pos)
            val2 = '%d/%d'%(v2,sum_neg)
            line_out = '\t'.join([key,str(val),val1,val2,str(round(v1/sum_pos,4)),
                                                             str(round(v2/sum_neg,4))])+'\n'
            outf.write(line_out)
    outf.close()

    return

def splitTrainData():
    from sklearn.model_selection import train_test_split
    import pandas as pd
    import numpy as np
    os.chdir('/home/minje/Projects/nlpfeatures')

    df = pd.read_csv('data/all-45.tsv',sep='\t')
    # df = pd.read_csv('data/all-345.tsv',sep='\t')
    labels = [x for x in df.columns if 'text' not in x]
    for label in labels:
        df2 = df[['text','h_text']]
        df2['label'] = np.array(df[label]>=2,dtype=int)
        df_tr, df_t = train_test_split(df2,test_size=0.2)
        df_t,df_v = train_test_split(df_t,test_size=0.5)
        df_tr.to_csv('data/xgboost-format/%s.train.tsv'%label,sep='\t',index=False)
        df_t.to_csv('data/xgboost-format/%s.test.tsv'%label,sep='\t',index=False)
        df_v.to_csv('data/xgboost-format/%s.valid.tsv'%label,sep='\t',index=False)
        print("Saved files for ",label)
    return

def saveToFlairFormat():
    import pandas as pd
    load_dir = '/home/minje/Projects/nlpfeatures/data/xgboost-format'
    save_dir = '/home/minje/Projects/nlpfeatures/data/flair-format'
    for filename in os.listdir(load_dir):
        df = pd.read_csv(join(load_dir,filename),sep='\t')
        # df['label'] = ['__label__'+str(x) for x in df['label'].values]
        with open(join(save_dir,filename),'w') as f:
            for line in df.values:
                f.write('__label__%s '%line[-1]+line[1].strip()+'\n')
    # print(df['label'])
    return

if __name__=='__main__':
    # splitTrainData()
    # word2vec4gensim('### File directory where the .bin or .vec file is located')
    # train_ngrams(file_dir='data/all-456.tsv',min_count=10)
    saveLogOddNGrams()
    # saveToFlairFormat()