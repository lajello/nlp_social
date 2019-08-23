# nlp_social
NLP tools for mining social exchange data

Installation steps

0 (optional) - create a new Conda environment

1 - type "conda install pip"

2 - type "pip install -r requirements.txt"

Running training models

1 - run "python models/train_xgboost.py" or "python models/train_pytorch.py"

Issues

1 - n-gram features may not work depending on the Gensim version.
In that case, please create your own n-grams using the train_ngrams function in preprocessing/preprocessing_functions.py

2 - pytorch LSTMs require pretrained word embeddings of .wv format.
Depending on the type of word embedding you wish to use, please transform the downloadable vectors into .wv format using the functions in preprocessing/preprocessing_functions.py
