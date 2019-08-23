import numpy as np

# get features for argumentation
def getArgumentFeatures(words):
    """
    from https://arxiv.org/pdf/1506.04744.pdf

    To quantify the level of argumentation, we
    calculate average number of claim and premise
    markers per sentence, as identified by Stab and
    Gurevych (Identifying argumentative discourse
    structures in persuasive essays, 2014).
    We also measure the number of
    request sentences in each message, as identified by
    the heuristics in the Stanford Politeness classifier
    (Danescu-Niculescu-Mizil et al., 2013).
    :param words:
    :return:
    """
    assert type(words)==list, "Input is not list of words"
    arr = np.zeros(10)

    return arr

# get features for discourse complexity
def getDiscourseFeatures(words):
    """
    from https://arxiv.org/pdf/1506.04744.pdf

    To quantify the level of argumentation, we
    calculate average number of claim and premise
    markers per sentence, as identified by Stab and
    Gurevych (2014).
    We also measure the number of
    request sentences in each message, as identified by
    the heuristics in the Stanford Politeness classifier
    (Danescu-Niculescu-Mizil et al., 2013).
    :param words:
    :return:
    """
    assert type(words) == list, "Input is not list of words"
    arr = np.zeros(10)

    return arr

def getSubjectivityFeatures(words):
    """
    from https://arxiv.org/pdf/1506.04744.pdf

    We explored phrases expressing
    opinion, accusation, suspicion, and speculation
    taken from an automatically collected lexicon
    (Riloff and Wiebe, 2003),
    :param words:
    :return:
    """

    return

if __name__=='__main__':
    words = ['this','is','a','sentence']
    getArgumentFeatures(words)