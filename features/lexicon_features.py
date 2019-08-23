"""
Features where a word or phrase belonging to a lexicon exists
"""

import os
from os.path import join
import pickle
import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion

class ExtractAllLinguistics(BaseEstimator, TransformerMixin):
    def __init__(self):
        with open('lexicons/concreteness.csv') as f:
            self.re_concrete = re.compile(r'\b(%s)\b'%'|'.join([x.split(',')[0] for x in f.read().strip().split('\n')]))
        with open('lexicons/differentiation_words.txt') as f:
            self.re_diff = re.compile(r'\b(%s)\b'%'|'.join([x.split(',')[0] for x in f.read().strip().split('\n')]))
        with open('lexicons/empathy.txt') as f:
            words = [x.split('\t')[0] for x in f.read().strip().split('\n')]
            self.re_empathy = re.compile(r'\b(%s)\b'%'|'.join([x for x in words if '*' not in x]))
            self.re_empathy2 = re.compile(r'\b(%s)\b'%'|'.join(['%s[a-z]+'%x[:-1] for x in words if '*' in x]))
        with open('lexicons/hedging_terms.csv') as f:
            self.re_hedging = re.compile(r'\b(%s)\b'%'|'.join([x.split(',')[0] for x in f.read().strip().split('\n')]))
        with open('lexicons/integration.txt') as f:
            self.re_integration = re.compile(r'\b(%s)\b'%'|'.join([x for x in f.read().strip().split('\n')]))
        with open('lexicons/misspellings.txt') as f:
            self.re_misspellings = re.compile(r'\b(%s)\b'%'|'.join([x.split('->')[0] for x in f.read().strip().split('\n')]))
        with open('lexicons/offensive_language.csv') as f:
            self.re_offensive = re.compile(r'\b(%s)\b'%'|'.join([x.split(',')[0] for x in f.read().strip().split('\n')]))

        return

    def fit(self, X):
        return

    # returns the number of concreteness-related words in a sentence
    def countWordsFromSentence(self, sentence):
        # sentence here should be a string; if not, make it a string
        if type(sentence)==list:
            sentence = ' '.join(sentence)

        out = np.zeros(6)

        # out[0] += len(self.re_concrete.findall(string=sentence))
        out[0] += len(self.re_diff.findall(string=sentence))
        out[1] += len(self.re_empathy.findall(string=sentence)) + len(self.re_empathy2.findall(string=sentence))
        out[2] += len(self.re_hedging.findall(string=sentence))
        out[3] += len(self.re_integration.findall(string=sentence))
        out[4] += len(self.re_misspellings.findall(string=sentence))
        out[5] += len(self.re_offensive.findall(string=sentence))

        return out

    def transform_single(self, sentences):
        out = np.zeros(6)
        for sent in sentences:
            out += self.countWordsFromSentence(sent)
        return out/max(1,len(sentences))

    def transform(self, X):
        return np.array([self.countWordsFromSentence(sents) for sents in X])

    def get_feature_names(self):
        return ['ling:'+x for x in ['differential','empathy','hedging','integration','misspellings','offensive']]


class ExtractNRC(BaseEstimator, TransformerMixin):
    """
    Features obtained from the NRC Emolex and VAD lexicons
    """
    def __init__(self):
        # emolex
        with open('lexicons/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt') as f:
            self.emo_words = {x:[] for x in range(10)}
            for ln,line in enumerate(f):
                line = line.strip().split('\t')
                if int(line[-1])==1:
                    self.emo_words[ln%10].append(line[0])
        self.re_emo = [re.compile(r'\b(%s)\b'%'|'.join(words)) for words in self.emo_words.values()]

        with open('lexicons/NRC-VAD-Lexicon.txt') as f:
            self.vad_words = {}
            for ln,line in enumerate(f):
                if ln==0:
                    continue
                line = line.strip().split('\t')
                self.vad_words[line[0]]=np.array([float(x) for x in line[1:]])

    def fit(self,X):
        return

    def transform_single(self,sentence):
        if type(sentence)==list:
            sentence = ' '.join(sentence)
        out1 = np.zeros(len(self.re_emo))
        for ln,reg in enumerate(self.re_emo):
            out1[ln]+=len(reg.findall(sentence))

        out2 = np.zeros(3)
        words = sentence.split()
        cnt = 0
        for word in words:
            if word in self.vad_words:
                out2+=self.vad_words[word]
                cnt+=1
        out2 = out2/max(cnt,1)

        out = out1.tolist()+out2.tolist()
        return out


    def transform(self,X):
        out = []
        for sent in X:
            out.append(self.transform_single(sent))
        return np.array(out)

    def get_feature_names(self):
        return ['nrc:'+x for x in ['anger','anticipation','disgust','fear','joy','negative','positive','sadness','surprise','trust'
                ,'v-score','a-score','d-score']]


# extract terms using the Empath client
class ExtractEmpath(BaseEstimator, TransformerMixin):
    def __init__(self):
        from empath import Empath
        self.lexicon = Empath()
        self.idx2cat = list(sorted(self.lexicon.cats.keys()))
        self.cat2idx = {k:ln for ln,k in enumerate(self.idx2cat)}
        return

    def fit(self, X):
        return

    # returns the number of empathy-related words in a sentence
    def get_empath_score_from_sentence(self, sentence):
        # sentence here should be a string; if not, make it a string
        out = np.zeros(len(self.cat2idx))
        if type(sentence)==list:
            sentence = ' '.join(sentence)
        sentence = sentence.lower()
        scores = self.lexicon.analyze(sentence, normalize=False)
        for i,cat in enumerate(self.idx2cat):
            if cat in scores:
                out[i] = scores[cat]
        return out

    def transform(self, X):
        out = []
        for sentence in X:
            out.append(self.get_empath_score_from_sentence(sentence))
        return np.array(out)

    def transform_single(self, sentence):
        arr = np.zeros(len(self.cat2idx))
        try:
            scores = self.get_empath_score_from_sentence(sentence)
            scores = np.array(scores>=1,dtype=int)
            arr+=scores
        except:
            pass
        return arr

    def get_feature_names(self):
        return ['empath:'+x for x in self.idx2cat]

# extracts the instances of emojis in a phrase
class ExtractEmoji(BaseEstimator, TransformerMixin):
    def __init__(self):
        with open('lexicons/emoji.txt') as f:
            self.idx2emoji = f.read().strip().split('\n')
        self.emoji2idx = {k:ln for ln,k in enumerate(self.idx2emoji)}
        return

    def fit(self, X):
        return

    # returns the number of concreteness-related words in a sentence
    def getEmojisFromSentence(self, sentence):
        arr = np.zeros(len(self.emoji2idx)+1)
        if (type(sentence)==list)&(type(sentence[0])==0):
            sentence = ' '.join(sentence)
        for c in sentence:
            if c in self.emoji2idx:
                arr[self.emoji2idx[c]]+=1
        arr[-1] = arr.sum()
        return arr

    def transform(self, X):
        out = []
        for sentences in X:
            out.append(self.transform_single(sentences))
        return np.array(out)

    def transform_single(self, sentences):
        arr = np.zeros(len(self.emoji2idx)+1)
        for sentence in sentences:
            arr += self.getEmojisFromSentence(sentence)
        arr = arr/max(len(sentences),1)
        return arr

    def get_feature_names(self):
        return ['emoji:'+x for x in self.idx2emoji]+['emoji:total']

# counts the number of hashtags in a phrase
class ExtractHashtags(BaseEstimator, TransformerMixin):
    def __init__(self):
        with open('lexicons/hashtags.txt') as f:
            self.idx2ht = f.read().strip().split('\n')
        self.ht2idx = {k:ln for ln,k in enumerate(self.idx2ht)}
        return

    def fit(self, X):
        return

    # returns an array of hashtags from an object
    def getHashtagsFromObject(self, obj):
        arr = np.zeros(len(self.ht2idx)+1)
        if 'hashtags' in obj:
            for ht in obj['hashtags']:
                ht = ht.lower()
                if not ht.startswith('#'):
                    ht = '#'+ht
                if ht in self.ht2idx:
                    arr[self.ht2idx[ht]] += 1
        arr[-1] = len(obj['hashtags']) # count of all hashtags, whether or not in our set
        return arr

    def transform(self, X):
        out = []
        for obj_list in X:
            out.append(self.transform_single(obj_list))
        return np.array(out)

    def transform_single(self, obj_list):
        arr = np.zeros(len(self.ht2idx)+1)
        for obj in obj_list:
            arr += self.getHashtagsFromObject(obj)
        arr = arr / max(len(obj_list),1)
        return arr

    def get_feature_names(self):
        return ['ht:'+x for x in self.idx2ht]+['ht:total']

# counts the distribution of hashtags in a phrase
class ExtractHashtagTopics(BaseEstimator, TransformerMixin):
    def __init__(self):
        import ujson as json
        with open('lexicons/hashtag2topic.json') as f:
            self.ht2idx = json.load(f)
            self.n_topics = len(set([v for v in self.ht2idx.values()]))
        return


    def fit(self, X):
        return

    # returns an array of hashtags from an object
    def getHashtagsFromObject(self, obj):
        arr = np.zeros(self.n_topics)
        if 'hashtags' in obj:
            for ht in obj['hashtags']:
                ht = ht.lower().replace('#','')
                if ht in self.ht2idx:
                    arr[self.ht2idx[ht]] += 1
        return arr

    def transform(self, X):
        out = []
        for obj_list in X:
            out.append(self.transform_single(obj_list))
        return np.array(out)

    def transform_single(self, obj_list):
        arr = np.zeros(self.n_topics)
        for obj in obj_list:
            arr += self.getHashtagsFromObject(obj)
        # arr = arr / max(len(obj_list),1)
        arr = arr / max(arr.sum(),1)
        return arr

    def get_feature_names(self):
        return ['ht-topic:%d'%i for i in range(self.n_topics)]

# extract hateness-related terms using the HateSonar client
class ExtractHateSonar(BaseEstimator, TransformerMixin):
    def __init__(self):
        from hatesonar import Sonar
        self.sonar = Sonar()
        return

    def fit(self, X):
        return

    # returns the number of empathy-related words in a sentence
    def get_hate_score_from_sentence(self, sentence):
        # sentence here should be a string; if not, make it a string
        if type(sentence)==list:
            sentence = ' '.join(sentence)
        out = self.sonar.ping(sentence)['classes']
        score = [x['confidence'] for x in out]
        return np.array(score)

    def transform(self, X):
        return np.array([self.get_hate_score_from_sentence(sent) for sent in X])
        """
        :param X: either a list of sentences(tokenized) or a list of (list of sentences(tokenized));
        the latter is when a user has several different sentences in a dataset
        :return:
        """

    def get_feature_names(self):
        return ['hate:'+x for x in ['hate_speech','offensive_language','neither']]

# extracts LIWC from multiple users: text should be tokenized but not n-gram-merged
class ExtractLIWC(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.load_liwc()

    def fit(self, X):
        return

    def load_liwc(self):
        """
        word2liwc: returns the dimensions that a word belongs to, uses original index (1-126)
        lidx2cat: returns the description of the dimension for each original index (1-126)
        lidx2vidx: maps sparse 126-dimension to a dense 71-dimension idx
        :return:
        """
        with open('lexicons/liwc.pckl', 'rb') as f:
            self.word2liwc = pickle.load(f)
        with open('lexicons/LIWC2015_English.dic') as f:
            liwc_idx = f.readlines()
        self.lidx2cat = dict()
        idx = 0
        self.lidx2vidx = {}  # LIWC index to vector index -> 126-dim to 71-dim or so
        for line in liwc_idx[0:77]:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            self.lidx2vidx[int(line[0])] = idx
            idx += 1
            self.lidx2cat[int(line[0])] = line[1].split()[0]
        self.vidx2lidx = {v: k for k, v in self.lidx2vidx.items()}
        self.vidx2cat = {self.lidx2vidx[idx]: cat for idx, cat in self.lidx2cat.items()}
        return

    def sentence2liwc(self, sentence):
        arr = np.zeros(len(self.lidx2vidx))
        for word in sentence:
            word = word.lower()
            if word in self.word2liwc:
                idxs = self.word2liwc[word]
                for idx in idxs:
                    arr[self.lidx2vidx[idx]]+=1
        return arr

    def get_feature_names(self):
        return ['liwc:%s' % k for k in self.vidx2cat.values()]

    def transform(self, X):
        return np.array([self.sentence2liwc(sent) for sent in X])

    def transform_single(self, sentences):
        arr = np.zeros(len(self.lidx2vidx))
        for words in sentences:
            for idx in set(self.sentence2liwc(words)):
                arr[idx]+=1
        arr = arr/max(len(sentences),1)
        return arr

# extract n-grams from tokenized text
class ExtractNGrams(BaseEstimator, TransformerMixin):
    def __init__(self,dimension=None):
        lex_dir = 'lexicons/'
        with open(join(lex_dir,'phraser-bi.pckl'), 'rb') as f:
            self.bi_phraser = pickle.load(f)
        with open(join(lex_dir,'phraser-tri.pckl'), 'rb') as f:
            self.tri_phraser = pickle.load(f)
        if dimension:
            with open(join(lex_dir,'vocab-%s.tsv'%dimension)) as f:
                self.vocab = [x.split()[0] for x in f.read().strip().split('\n')]
        else:
            with open(join(lex_dir,'ngram-vocab.txt')) as f:
                self.vocab = f.read().strip().split('\n')

        self.ngram2idx = {k: i for i, k in enumerate(self.vocab)}
        self.idx2ngram = [k for k in self.ngram2idx.keys()]

    def fit(self, X):
        return

    def get_feature_names(self):
        return ['ngram:%s' % k for k in self.ngram2idx.keys()]

    def getNgramsFromSentence(self, words):  # sentence is a list of words here
        arr = np.zeros(len(self.idx2ngram))
        if type(words)==str:
            words = words.split()
        words = [x.lower() for x in words]
        for phrase in self.tri_phraser[self.bi_phraser[words]]:
            if phrase in self.ngram2idx:
                arr[self.ngram2idx[phrase]]=1
        return arr

    def transform_single(self, X):
        out = np.zeros(len(self.idx2ngram))
        for sent in X:
            out += self.getNgramsFromSentence(sent)
        out = out / max(len(X),1)
        return out

    def transform(self, X):
        """
        :param X: a list of tokenized sentences
        :return: arr
        """
        if len(X)==0:
            return np.zeros(len(self.idx2ngram))
        else:
            return np.array([self.getNgramsFromSentence(sent) for sent in X])


# extract concrete-related words from tokenized text
class ExtractVader(BaseEstimator, TransformerMixin):
    def __init__(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self.vader = SentimentIntensityAnalyzer().polarity_scores
        return

    def fit(self, X):
        return

    # returns the number of concreteness-related words in a sentence
    def getVaderScoreFromSentence(self, sentence):
        # sentence here should be a string; if not, make it a string
        if type(sentence)==list:
            sentence = ' '.join(sentence)
        scores = self.vader(sentence)
        arr = np.array([scores[x] for x in ['pos','neu','neg']])
        return arr

    def transform(self, X):
        """
        :param X: either a list of sentences(tokenized) or a list of (list of sentences(tokenized));
        the latter is when a user has several different sentences in a dataset
        :return:
        """
        if len(X)==0:
            return np.zeros(4)
        else:
            return np.array([self.getVaderScoreFromSentence(sent) for sent in X])

    def get_feature_names(self):
        return ['vader:'+x for x in ['pos','neu','neg']]

if __name__=='__main__':
    print(os.getcwd())
    os.chdir('../')
    print(os.getcwd())
    from nltk.tokenize import WordPunctTokenizer
    tok = WordPunctTokenizer().tokenize
    lines = []
    with open('data/sample.txt') as f:
        for line in f:
            lines.append(tok(line.lower()))



    # ex = ExtractLIWC()
    # words = 'this is my castle'.split()
    # print(ex.transform_single(words))

    # ex = ExtractNGrams()
    # from nltk.tokenize import WordPunctTokenizer
    # tok = WordPunctTokenizer().tokenize
    # with open('/home/minje/Projects/nlpfeatures/data/sample.txt') as f:
    #     text = f.read()[:500]
    # words = tok(text.lower())
    # # words = 'hyped up for the world cup'.split()
    # print(ex.phraser[words])
    # idxs = ex.get_ngrams_from_sentence(words)
    # for idx in idxs:
    #     print(idx,ex.idx2ngram[idx])

    # ex = ExtractMisspellings()
    # words = 'my aera is getting small thank yuo'
    # out = ex.get_misspelling_from_sentence(words.split())
    # print(out)

    # ex = ExtractDifferentiation()
    # words = 'alternatively, this does not sound appropriate too'
    # out = ex.count_diff_words_from_sentence(words.split())
    # print(out)

    # ex = ExtractEmpathy()
    # words = 'devoting donating'
    # out = ex.count_empathy_words_from_sentence(words.split())
    # print(out)

    # ex = ExtractHedging()
    # words = 'that is not unreasonable'
    # out = ex.count_hedge_terms_from_sentence(words.split())
    # print(out)

    # ex = ExtractIntegration()
    # words = 'there is a tradeoff between this and that'
    # out = ex.count_integration_words_from_sentence(words.split())
    # print(out)

    # ex = ExtractConcreteness()
    # words = 'that was an abnormal act , lead to abortion too'
    # out = ex.count_concrete_words_from_sentence(words.split())
    # print(out)

    # ex = ExtractEmpath()
    # words = 'he hit the other person'
    # out = ex.get_empath_score_from_sentence(words.split())
    # print(out)
    # print(len(out))


    p = FeatureUnion([
        ('ex1',ExtractMisspellings()),
        ('ex2',ExtractOffensive()),
        ('ex3',ExtractConcreteness()),
        ('ex4',ExtractIntegration()),
        ('ex5',ExtractHedging()),
        ('ex6',ExtractEmpathy()),
        # ('ex7',ExtractLIWC()),
        ('ex8',ExtractNGrams()),
        ('ex9',ExtractHateSonar()),
        ('ex10',ExtractEmpath()),
        ('ex11',ExtractVader()),
    ])

