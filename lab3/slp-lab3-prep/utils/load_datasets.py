import glob
import html
import os

import numpy as np 

from config import DATA_PATH

SEPARATOR = "\t"


def clean_text(text):
    """
    Remove extra quotes from text files and html entities
    Args:
        text (str): a string of text

    Returns: (str): the "cleaned" text

    """
    text = text.rstrip()

    if '""' in text:
        if text[0] == text[-1] == '"':
            text = text[1:-1]
        text = text.replace('\\""', '"')
        text = text.replace('""', '"')

    text = text.replace('\\""', '"')

    text = html.unescape(text)
    text = ' '.join(text.split())
    return text


def parse_file(file):
    """
    Read a file and return a dictionary of the data, in the format:
    tweet_id:{sentiment, text}
    """

    data = {}
    lines = open(file, "r", encoding="utf-8").readlines()
    for line_id, line in enumerate(lines):
        columns = line.rstrip().split(SEPARATOR)
        tweet_id = columns[0]
        sentiment = columns[1]
        text = columns[2:]
        text = clean_text(" ".join(text))
        data[tweet_id] = (sentiment, text)
    return data


def load_from_dir(path):
    files = glob.glob(path + "/**/*.tsv", recursive=True)
    files.extend(glob.glob(path + "/**/*.txt", recursive=True))

    data = {}  # use dict, in order to avoid having duplicate tweets (same id)
    for file in files:
        file_data = parse_file(file)
        data.update(file_data)
    return list(data.values())


def load_Semeval2017A():
    train = load_from_dir(os.path.join(DATA_PATH, "Semeval2017A/train_dev"))
    test = load_from_dir(os.path.join(DATA_PATH, "Semeval2017A/gold"))

    X_train = [x[1] for x in train]
    y_train = [x[0] for x in train]
    X_test = [x[1] for x in test]
    y_test = [x[0] for x in test]

    return X_train, y_train, X_test, y_test


def load_MR():
    pos = open(os.path.join(DATA_PATH, "MR/rt-polarity.pos")).readlines()
    neg = open(os.path.join(DATA_PATH, "MR/rt-polarity.neg")).readlines()

    pos = [x.strip() for x in pos]
    neg = [x.strip() for x in neg]

    pos_labels = ["positive"] * len(pos)
    neg_labels = ["negative"] * len(neg)

    split = 5000

    X_train = pos[:split] + neg[:split]
    y_train = pos_labels[:split] + neg_labels[:split]

    X_test = pos[split:] + neg[split:]
    y_test = pos_labels[split:] + neg_labels[split:]

    return X_train, y_train, X_test, y_test

def load_EI_Reg():
    '''
    Read data and return in dictionary with dataset-emotion key
    Code is self-explanatory  
    '''
    X = {}
    y = {}
    name = {'development' : 'dev',
            'training' : 'train',
            'test-gold' : 'test-gold'
            }

    for emotion in ['joy','anger','fear','sadness']:

        for dataset in ['development','training','test-gold']:

            path = os.path.join(DATA_PATH, f"SemEval2018-Task1-all-data/English/EI-reg/{dataset}/2018-EI-reg-En-{emotion}-{name[dataset]}.txt")
            score,data = parse_file_regression(path)

            X[f'{name[dataset]}-{emotion}'] = data 
            y[f'{name[dataset]}-{emotion}'] = score
    
    return X,y 

def parse_file_regression(file):
    """
    Read a file and return a dictionary of the data, in the format:
    tweet_id:{sentiment, text}
    """

    data = []
    scores = []

    lines = open(file, "r", encoding="utf-8").readlines()
    for line_id, line in enumerate(lines[1:]):
        columns = line.rstrip().split(SEPARATOR)
        
        text = columns[1]
        score = columns[3]
        
        scores.append(score)
        data.append(text)
    scores = np.array(scores)
    data = np.array(data)

    return (score,data)
    

