"""
Code from https://medium.com/swlh/a-simple-guide-on-using-bert-for-text-classification-bbf041ac8d04

Instructions are available in the medium blog. This code is to make it run in a Python script
"""

from __future__ import absolute_import, division, print_function
from os.path import join
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from sklearn.metrics import average_precision_score as AP
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import accuracy_score as ACC
from sklearn.metrics import recall_score as REC
from sklearn.metrics import precision_score as PRE


import csv
import os
import sys
import logging

logger = logging.getLogger()
csv.field_size_limit(2147483647) # Increase CSV reader's field limit incase we have long text.

import pickle
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,TensorDataset)
from torch.nn import CrossEntropyLoss, MSELoss
from tqdm import tqdm_notebook, trange
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from multiprocessing import Pool, cpu_count


# saves a file to fit the BERT format (would like to know how to use dev dataset as well)
def saveData(dim):
    save_dir = join('/home/minje/Projects/nlpfeatures/data/bert-format',dim)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if os.path.exists(join(save_dir,'train.tsv')) and \
        os.path.exists(join(save_dir,'test.tsv')) and \
        os.path.exists(join(save_dir,'dev.tsv')):
        return
    else:
        sys.path.append('/home/minje/Projects/nlpfeatures')
        from nltk.tokenize import TweetTokenizer
        from sklearn.model_selection import train_test_split
        from preprocessing.customDataLoader import preprocessText
        tokenize = TweetTokenizer().tokenize

        df = pd.read_csv('/home/minje/Projects/nlpfeatures/data/all-456.tsv', sep='\t')
        dim_col_idx = df.columns.tolist().index(dim)
        text_col_idx = df.columns.tolist().index('h_text')
        X = []
        y = []
        # append positive/negative samples to X and y lists
        for line in df.values:
            label = line[dim_col_idx]
            if (label >= 2) | (label == 0):
                text = preprocessText(line[text_col_idx])
                X.append(text)
                y.append(int(label >= 2))

        X_tr, X_t, y_tr, y_t = train_test_split(X, y, test_size=0.2, random_state=42)
        X_t, X_v, y_t, y_v = train_test_split(X_t, y_t, test_size=0.5, random_state=42)

        # save to tsv files
        out = []
        for i,(X_,y_) in enumerate(zip(X_tr,y_tr)):
            out.append((y_,'a',X_))
        df = pd.DataFrame(out)
        df.to_csv(join(save_dir,'train.tsv'),sep='\t',header=None)

        out = []
        for i,(X_,y_) in enumerate(zip(X_t,y_t)):
            out.append((y_,'a',X_))
        df = pd.DataFrame(out)
        df.to_csv(join(save_dir,'test.tsv'),sep='\t',header=None)

        out = []
        for i,(X_,y_) in enumerate(zip(X_v,y_v)):
            out.append((y_,'a',X_))
        df = pd.DataFrame(out)
        df.to_csv(join(save_dir,'dev.tsv'),sep='\t',header=None)
        return

def train(dim):
    import logging
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    code_dir = '### set to directory where you have the model code'
    os.chdir('/home/minje/Projects/nlpfeatures/')
    DATA_DIR = "data/bert-format/%s"%dim
    BERT_MODEL = 'bert-base-cased'
    TASK_NAME = '%s-prediction'%dim

    # The output directory where the fine-tuned model and checkpoints will be written.
    OUTPUT_DIR = f'results/performance/BERT/{TASK_NAME}/'
    # The directory where the evaluation reports will be written to.
    REPORTS_DIR = f'results/performance/BERT/{TASK_NAME}_evaluation_report/'
    # This is where BERT will look for pre-trained models to load parameters from.
    CACHE_DIR = 'cache/'

    # The maximum total input sequence length after WordPiece tokenization.
    # Sequences longer than this will be truncated, and sequences shorter than this will be padded.
    MAX_SEQ_LENGTH = 20

    TRAIN_BATCH_SIZE = 24
    EVAL_BATCH_SIZE = 8
    LEARNING_RATE = 2e-5
    NUM_TRAIN_EPOCHS = 4
    RANDOM_SEED = 42
    GRADIENT_ACCUMULATION_STEPS = 1
    WARMUP_PROPORTION = 0.1
    OUTPUT_MODE = 'classification'
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"

    if os.path.exists(REPORTS_DIR) and os.listdir(REPORTS_DIR):
        REPORTS_DIR += f'/report_{len(os.listdir(REPORTS_DIR))}'
        os.makedirs(REPORTS_DIR)
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)
        REPORTS_DIR += f'/report_{len(os.listdir(REPORTS_DIR))}'
        os.makedirs(REPORTS_DIR)
    # if os.path.exists(OUTPUT_DIR) and os.listdir(OUTPUT_DIR):
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(OUTPUT_DIR))
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    processor = BinaryClassificationProcessor()
    train_examples = processor.get_train_examples(DATA_DIR)
    train_examples_len = len(train_examples)
    label_list = processor.get_labels()  # [0, 1] for binary classification
    num_labels = len(label_list)
    num_train_optimization_steps = int(
        train_examples_len / TRAIN_BATCH_SIZE / GRADIENT_ACCUMULATION_STEPS) * NUM_TRAIN_EPOCHS


    if os.path.exists(join(DATA_DIR,"train_features.pckl")):
        with open(join(DATA_DIR,"train_features.pkl"), "rb") as f:
            train_features = pickle.load(f)
    else:
        # Load pre-trained model tokenizer (vocabulary)
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
        label_map = {label: i for i, label in enumerate(label_list)}
        train_examples_for_processing = [(example, label_map, MAX_SEQ_LENGTH, tokenizer, OUTPUT_MODE) for example in
                                         train_examples]
        process_count = cpu_count() - 1
        process_count = 30
        if __name__ ==  '__main__':
            print(f'Preparing to convert {train_examples_len} examples..')
            print(f'Spawning {process_count} processes..')
            with Pool(process_count) as p:
                train_features = list(tqdm_notebook(p.imap(convert_example_to_feature, train_examples_for_processing), total=train_examples_len))
        with open(join(DATA_DIR,"train_features.pkl"), "wb") as f:
            pickle.dump(train_features,f)

    model = BertForSequenceClassification.from_pretrained(BERT_MODEL, cache_dir=CACHE_DIR, num_labels=num_labels)
    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=LEARNING_RATE,
                         warmup=WARMUP_PROPORTION,
                         t_total=num_train_optimization_steps)
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", train_examples_len)
    logger.info("  Batch size = %d", TRAIN_BATCH_SIZE)
    logger.info("  Num steps = %d", num_train_optimization_steps)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

    if OUTPUT_MODE == "classification":
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    elif OUTPUT_MODE == "regression":
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=TRAIN_BATCH_SIZE)

    model.train()
    for _ in trange(int(NUM_TRAIN_EPOCHS), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm_notebook(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            logits = model(input_ids, segment_ids, input_mask, labels=None)

            if OUTPUT_MODE == "classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
            elif OUTPUT_MODE == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), label_ids.view(-1))

            if GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / GRADIENT_ACCUMULATION_STEPS

            loss.backward()
            print("\r%f" % loss, end='')

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(OUTPUT_DIR, WEIGHTS_NAME)
    output_config_file = os.path.join(OUTPUT_DIR, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(OUTPUT_DIR)

    return

def test(dim):
    import logging
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.chdir('/home/minje/Projects/nlpfeatures/')
    DATA_DIR = "data/bert-format/%s"%dim
    TASK_NAME = '%s-prediction'%dim

    OUTPUT_DIR = f'results/performance/BERT/{TASK_NAME}/'
    REPORTS_DIR = f'results/performance/BERT/{TASK_NAME}_evaluation_report/'
    CACHE_DIR = 'cache/'

    # The maximum total input sequence length after WordPiece tokenization.
    # Sequences longer than this will be truncated, and sequences shorter than this will be padded.
    MAX_SEQ_LENGTH = 20
    EVAL_BATCH_SIZE = 8
    OUTPUT_MODE = 'classification'

    if os.path.exists(REPORTS_DIR) and os.listdir(REPORTS_DIR):
        REPORTS_DIR += f'/report_{len(os.listdir(REPORTS_DIR))}'
        os.makedirs(REPORTS_DIR)
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)
        REPORTS_DIR += f'/report_{len(os.listdir(REPORTS_DIR))}'
        os.makedirs(REPORTS_DIR)
    # if os.path.exists(OUTPUT_DIR) and os.listdir(OUTPUT_DIR):
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(OUTPUT_DIR))
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    tokenizer = BertTokenizer.from_pretrained(OUTPUT_DIR + 'vocab.txt', do_lower_case=False)
    processor = BinaryClassificationProcessor()
    eval_examples = processor.get_dev_examples(DATA_DIR)
    label_list = processor.get_labels()  # [0, 1] for binary classification
    num_labels = len(label_list)
    eval_examples_len = len(eval_examples)

    label_map = {label: i for i, label in enumerate(label_list)}
    eval_examples_for_processing = [(example, label_map, MAX_SEQ_LENGTH, tokenizer, OUTPUT_MODE) for example in
                                    eval_examples]
    process_count = 30
    if __name__ == '__main__':
        print(f'Preparing to convert {eval_examples_len} examples..')
        print(f'Spawning {process_count} processes..')
        with Pool(process_count) as p:
            eval_features = list(tqdm_notebook(
                p.imap(convert_example_to_feature, eval_examples_for_processing),
                total=eval_examples_len))

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

    if OUTPUT_MODE == "classification":
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    elif OUTPUT_MODE == "regression":
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=EVAL_BATCH_SIZE)
    # Load pre-trained model (weights)
    model = BertForSequenceClassification.from_pretrained(OUTPUT_DIR, cache_dir=CACHE_DIR, num_labels=len(label_list))
    model.to(device)
    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    y_true = []
    y_prob = []
    for input_ids, input_mask, segment_ids, label_ids in tqdm_notebook(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        y_true.extend(label_ids.squeeze().cpu().tolist())

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)

        # create eval loss and other metric required by the task
        if OUTPUT_MODE == "classification":
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        elif OUTPUT_MODE == "regression":
            loss_fct = MSELoss()
            tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = preds[0]
    if OUTPUT_MODE == "classification":
        probs = preds[:, 1].tolist()
        y_prob.extend(probs)
        preds = np.argmax(preds, axis=1)
    elif OUTPUT_MODE == "regression":
        preds = np.squeeze(preds)
    result = compute_metrics(TASK_NAME, all_label_ids.numpy(), preds)

    result['eval_loss'] = eval_loss

    output_eval_file = os.path.join(REPORTS_DIR, "eval_results.txt")
    output_eval_file = output_eval_file.replace('//', '/')
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key in (result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    y_pred = preds.tolist()

    acc = ACC(y_true,y_pred)
    pre = PRE(y_true,y_pred)
    rec = REC(y_true,y_pred)
    auc = AUC(y_true, y_prob)
    ap = AP(y_true, y_prob)
    print("AUC: %1.3f"%auc)
    print("AUCPR: %1.3f"%ap)

    save_dir='results/performance/BERT/%s'%TASK_NAME
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(join(save_dir, 'BERT-%s-scores.tsv' % (dim)), 'w') as f:
        f.write('\t'.join(['AUC','REC','PRE','ACC','AP'])+'\n')
        f.write('\t'.join([str(auc), str(rec), str(pre), str(acc), str(ap)]) + '\n')
    return


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class BinaryClassificationProcessor(DataProcessor):
    """Processor for binary classification dataset."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def convert_example_to_feature(example_row):
    # return example_row
    example, label_map, max_seq_length, tokenizer, output_mode = example_row

    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    if tokens_b:
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    if output_mode == "classification":
        label_id = label_map[example.label]
    elif output_mode == "regression":
        label_id = float(example.label)
    else:
        raise KeyError(output_mode)

    return InputFeatures(input_ids=input_ids,
                         input_mask=input_mask,
                         segment_ids=segment_ids,
                         label_id=label_id)

def get_eval_report(task_name, labels, preds):
    mcc = matthews_corrcoef(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    return {
        "task": task_name,
        "mcc": mcc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }

def compute_metrics(task_name, labels, preds):
    assert len(preds) == len(labels)
    return get_eval_report(task_name, labels, preds)


if __name__=='__main__':
    # code for saving text to bert-friendly form

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
        # saveData(dim)

        train(dim)
        test(dim)





