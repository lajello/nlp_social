"""
List of features from readability scores: should be in original text form, untokenized
"""
import numpy as np
import readability
import textstat
from sklearn.base import BaseEstimator, TransformerMixin

# get overall features using readability API (faster than textstat)
# def getReadability(text):

"""
A class that uses the readability API to obtain readability scores as well as other metrics such as 
the average number of characters in a word.
Note that to use the readability API, each sentence should be divided with a '\n' delimiter.
"""
# extract readability features from
class ExtractReadability(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feat_list = ['Kincaid', 'ARI', 'Coleman-Liau', 'FleschReadingEase', 'GunningFogIndex', 'LIX',
                          'SMOGIndex', 'RIX', 'DaleChallIndex',
                          'characters_per_word', 'syll_per_word',
                          'words_per_sentence', 'type_token_ratio']
        self.feat_set = set(self.feat_list)

    def get_readability_features_from_sentence(self, sentence):
        # a paragraph here is a
        if (type(sentence)==list) & (type(sentence[0])==str):
            sentence = ' '.join(sentence)
        features = readability.getmeasures(text=sentence, lang='en')
        out = []
        # get features from readability grades
        for cat in ['readability grades', 'sentence info']:
            for k, v in features[cat].items():
                if k in self.feat_set:
                    out.append(v)
        return np.array(out)

    def get_feature_names(self):
        return ['read:%s' % k for k in self.feat_list]

    def transform(self, X):
        out = []
        for sent in X:
            try:
                out.append(self.get_readability_features_from_sentence(sent))
            except:
                out.append(np.zeros(len(self.feat_list)))
        return np.array(out)
        # return np.array([self.get_readability_features_from_sentence(sent) for sent in X])
        # assert type(X) == list, "Error in ExtractReadability: input is not a list!"
        # assert type(X[0]) == list, "Error in ExtractReadability: Input is not tokenized!"
        # out = []
        # # case 1: only 1 sentence per sample
        # if type(X[0][0]) == str:
        #     for sentence in X:
        #         arr = self.get_readability_features_from_sentence(sentence)  # sentence = list of words
        #         out.append(arr)
        # # case 2: each sample has multiple sentences
        # elif type(X[0][0]) == list:
        #     if type(X[0][0][0]) == str:
        #         for sentences in X:
        #             arr = np.zeros(len(self.feat_list))
        #             for sentence in sentences:
        #                 arr += self.get_readability_features_from_sentence(sentence)
        #             out.append(arr/len(sentences))
        # return out



# get Flesch-Kincaid grade for reading test
def getFleschKincaid(text):
    assert type(text)==str,"Input is not a string"
    return textstat.flesch_kincaid_grade(text)

# get LIX score

# get SMOG grade
def getSMOG(text):
    assert type(text)==str,"Input is not a string"
    return textstat.smog_index(text)

# get Coleman-Liau index
def getColemanLiau(text):
    assert type(text)==str,"Input is not a string"
    return textstat.coleman_liau_index(text)

# get automated readability index
def getAutomatedReadability(text):
    assert type(text)==str,"Input is not a string"
    return textstat.automated_readability_index(text)

# get Gunning-Fog index
def getGunningFog(text):
    assert type(text)==str,"Input is not a string"
    return textstat.gunning_fog(text)


if __name__=='__main__':
    text = ("Playing games has always been thought to be important to "
    "the development of well-balanced and creative children; "
    "however, what part, if any, they should play in the lives "
    "of adults has never been researched that deeply. I believe "
    "that playing games is every bit as important for adults "
    "as for children. Not only is taking time out to play games "
    "with our children and other adults valuable to building "
    "interpersonal relationships but is also a wonderful way "
    "to release built up tension.")

    ex = ExtractReadability()
    out = ex.get_readability_features_from_string(text)
    print(out)
    print(ex.get_feature_names())
    # text = 'Holy cow! I dunno what to do'
    # print(getAutomatedReadability(text))
    # print(getGunningFog(text))
    # print(getColemanLiau(text))
    # print(getFleschKincaid(text))
    # print(getSMOG(text))