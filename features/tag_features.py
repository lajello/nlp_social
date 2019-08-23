"""
Deals with POS-tagging, dependency parsing, NER
"""

import spacy
from sklearn.base import BaseEstimator,TransformerMixin

# extract POS tags
class ExtractTags(BaseEstimator, TransformerMixin):
    def __init__(self):
        # load spacy model
        import spacy
        self.nlp = spacy.load('en_core_web_sm')
        self.idx2pos = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'CONJ', 'DET', 'EOL', 'IDS', 'INTJ', 'NAMES', 'NOUN', 'NO_TAG',
                    'NUM',
                    'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SPACE', 'SYM', 'VERB', 'X']
        self.pos2idx = {k: idx for idx, k in enumerate(self.idx2pos)}

    def fit(self, X):
        return

    def get_feature_names(self):
        return ['pos:%s' % k for k in self.pos2idx.keys()]

    def get_tags_from_string(self, words):
        sentence = ' '.join(words)
        tagged = self.nlp(sentence)
        arr = np.zeros(len(self.pos2idx))
        for tok in tagged:
            arr[self.pos2idx[tok.pos_]] += 1
        if arr.sum() > 0:
            arr = arr / arr.sum()
        return arr

    def transform(self, X):
        assert type(X) == list, "Error in ExtractTags: input is not a list!"
        assert type(X[0]) == list, "Error in ExtractTags: Input is not tokenized!"
        out = []
        # case 1: only 1 sentence per sample
        if type(X[0][0]) == str:
            for sentence in X:
                arr = self.get_tags_from_string(sentence)  # sentence = list of words
                out.append(arr)
        # case 2: each sample has multiple sentences
        elif type(X[0][0]) == list:
            if type(X[0][0][0]) == str:
                for sentences in X:
                    arr = np.zeros(len(self.idx2pos))
                    for sentence in sentences:
                        arr += self.get_tags_from_string(sentence)
                    out.append(arr/len(sentences))
        return out


class SpacyModel(object):
    def __init__(self):
        # load spacy model
        self.nlp = spacy.load('en')
        pos_list = ['ADJ','ADP','ADV','AUX','CCONJ','CONJ','DET','EOL','IDS','INTJ','NAMES','NOUN','NO_TAG','NUM',
                    'PART','PRON','PROPN','PUNCT','SCONJ','SPACE','SYM','VERB','X']
        self.pos2idx = {k:idx for idx,k in enumerate(pos_list)}

    def process_text(self,text,pos=True,dep=True,ner=True):
        if not (pos|dep|ner):
            print("None enabled: returning empty value")
            return []
        out = {}
        if pos:
            out['pos']=[]
        if dep:
            out['dep']=[]
        if ner:
            out['ner']=[]
        doc = self.nlp(text)
        for token in doc:
            if pos:
                out['pos'].append(token.pos_)
            if dep:
                out['dep'].append(token.dep_)
        if ner:
            for ent in doc.ents:
                out['ner'].append((ent.text,ent.start_char,ent.end_char,ent.label_))
        return out


    # def
"""
NER: spacy is fastest, Stanford's old API is really slow but twice more accurate
https://towardsdatascience.com/a-comparison-between-spacy-ner-stanford-ner-using-all-us-city-names-c4b6a547290
"""

# from stanfordcorenlp import StanfordCoreNLP
import numpy as np

class CoreNLP(object):
    def __init__(self, url='http://localhost:9000'):
        from nltk.parse import CoreNLPParser,CoreNLPDependencyParser
        self.parser = CoreNLPParser(url=url)
        self.pos_tagger = CoreNLPParser(url=url, tagtype='pos')
        self.ner_tagger = CoreNLPParser(url=url, tagtype='ner')
        self.dep_parser = CoreNLPDependencyParser(url=url)

    # return pos-tags to text sentence
    def pos_tag_text(self,text):
        assert (type(text)==str)|(type(text)==list),"Neither string nor list of words"
        if type(text)==str:
            text = text.split()
        return list(self.pos_tagger.tag(text))

    # return dependency parsed text sentence
    def dep_parse_text(self,text):
        assert (type(text)==str)|(type(text)==list),"Neither string nor list of words"
        if type(text)==str:
            text = text.split()
        return list(self.dep_parser.parse(text))



"""

To start the Stanford model, start the server while in the stanford-corenlp~~ dir
(I saved it in libraries)
https://github.com/nltk/nltk/wiki/Stanford-CoreNLP-API-in-NLTK
run the following line to start the server:

java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
-preload tokenize,ssplit,pos,lemma,ner,parse,depparse \
-status_port 9000 -port 9000 -timeout 15000 & 

"""

if __name__=='__main__':
    mdl = ExtractTags()
    text = ['this is the sentence we were all waiting for']
    print(mdl.transform(text))
    print(mdl.get_feature_names())
    # sp = SpacyModel()
    # # sn = CoreNLP()
    # text = "@user can't believe just announced that they will not be starring for the Game of Thrones series."
    # from time import time
    # start = time()
    # print(sp.process_text(text))
    # print(time()-start)
    # start = time()
    # print(sn.pos_tag_text(text))
    # print(time()-start)
    # start = time()
    # print(sn.dep_parse_text(text))
    # print(time()-start)
