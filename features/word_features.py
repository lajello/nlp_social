import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

"""
A set of features that work on word-level
All inputs should be provided as a list of words, or a list of sentences (=list of words)
"""

# extract other basic word features from tokenized text
class ExtractWordFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        return

    def fit(self, X):
        return

    def get_feature_names(self):
        feats = ['capital', 'elongation', '?', '!', '...']
        return ['word:%s' % k for k in feats]

    # returns the number of capitalized words in a sentence
    def containsCapital(self, sentence, binary=False):
        if type(sentence)==list:
            sentence = ' '.join(sentence)
        # assert type(words) == list, "Input is not a list of words: received type %s instead"%str(type(words))
        # assert type(words[0]) == str, "Input is not a list of words: received type %s instead"%str(type(words[0]))
        result = re.findall(pattern=r'(([A-Z]+){2,}\b)', string=sentence)
        if result:
            if binary:
                return 1
            else:
                return len(result)
        else:
            return 0

    # returns the number of elongated words or punctuations in a sentence
    def containsElongation(self, sentence, binary=False):
        if type(sentence)==list:
            sentence = ' '.join(sentence)
        result = re.findall(pattern=r'([A-Za-z!?])\1{2}', string=sentence)
        if result:
            if binary:
                return 1
            else:
                return len(result)
        else:
            return 0

    # returns the number of question marks in a sentence
    def containsQuestionMark(self, sentence, binary=False):
        # binary = True when you want only the existence of a question mark, False when you want the total count
        if type(sentence)==list:
            sentence = ' '.join(sentence)
        if binary:
            return min(sentence.count('?'), 1)
        else:
            return sentence.count('?')

    # returns the number of exclamation marks in a sentence
    def containsExclamationMark(self, sentence, binary=False):
        if type(sentence)==list:
            sentence = ' '.join(sentence)
        # assert type(words) == list, "Input is not a list of words"
        # assert type(words[0]) == str, "Input is not a list of words"
        if binary:
            return min(sentence.count('!'), 1)
        else:
            return sentence.count('!')

    # returns the number of exclamation marks in a sentence
    def containsEllipsis(self, words, binary=False):
        # binary = True when you want only the existence of a mark, False when you want the total count
        if type(words)==str:
            words = words.split()
        cnt = 0
        for word in words:
            if '..' in word:
                if binary:
                    return 1
                cnt += 1
        return cnt

    def getFeaturesFromSentence(self, sentence, binary=False):  # sentence: tokenized, not lowered
        out = [self.containsCapital(sentence, binary),
               self.containsElongation(sentence, binary),
               self.containsQuestionMark(sentence, binary),
               self.containsExclamationMark(sentence, binary),
               self.containsEllipsis(sentence, binary)]
        return np.array(out)

    def transform_single(self, sentences, binary=True):
        out = np.zeros(5)
        for sent in sentences:
            out += self.getFeaturesFromSentence(sent, binary)
        out = out / max(len(sentences), 1)
        return out

    def transform(self, X):
        return np.array([self.getFeaturesFromSentence(sent,False) for sent in X])





if __name__=='__main__':
    # r = getWordLength(['holy','cow','??','!'])
    sentence = 'tom is is a bad tomboy... he is tooooooooo much a pain in the ass he is my girlfriend???'
    words = sentence.split()
    ex = ExtractWordFeatures()
    out = ex.transform([[words]])
    print(out)
    print(ex.get_feature_names())

    # r = getWordEntropy(words=words)
    # r = getNumberOfSyllables(words=words)
    # r = getNumberOfWords(words=words)
    # containsURL(['nope'])
    # containsURL(['this','is','https://www.naver.com','http://www.google.com'])
