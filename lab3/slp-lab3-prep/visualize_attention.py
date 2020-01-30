import os
import warnings
import sys

import json

import torch
from torch.utils.data import DataLoader
from models import BaselineDNN,BaseLSTM

from sklearn.preprocessing import LabelEncoder

from config import EMB_PATH
from dataloading import SentenceDataset

from utils.load_datasets import load_Semeval2017A
from utils.load_embeddings import load_word_vectors


torch.multiprocessing.set_sharing_strategy('file_system')




MAX_SEQ_LEN = 60
EMB_DIM = 50
EMBEDDINGS = os.path.join(EMB_PATH, "glove.twitter.27B.50d.txt")
BATCH_SIZE = 1 # for simple predictions

word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM)

_,_, X_test, y_test = load_Semeval2017A()



le = LabelEncoder()
le.fit(list(set(y_test)))
y_test = le.transform(y_test)

n_classes = len(list(le.classes_))

test_set = SentenceDataset(X_test, y_test, word2idx,MAX_SEQ_LEN)



test_loader = DataLoader(test_set, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=4)


model = BaseLSTM(output_size=3,  
                    embeddings=embeddings,
                    method = 'attention',
                    attention_size=MAX_SEQ_LEN,
                    bidirectional= False, # Don't Forget to also try without Bidirectional...
                    trainable_emb=False, 
                    return_weights=True)

checkpoint = torch.load(f'./checkpoints/Semeval2017A/LSTM_attention')

path = './reports/Semeval2017A/NEATVIS_LSTM_attention.json'
# Don't forget to change path for Bidirectional..


'''

model = BaselineDNN(output_size=3,  # EX8
                embeddings=embeddings,
                method = 'attention',
                attention_size=MAX_SEQ_LEN,
                trainable_emb=False,
                return_weights=True)

checkpoint = torch.load(f'./checkpoints/Semeval2017A/DNN_attention')
path = './reports/Semeval2017A/NEATVIS_DNN_attention.json'
'''
model.load_state_dict(checkpoint['model'])

model.eval()
samples_list = []

with torch.no_grad():
    for i,batch in enumerate(test_loader):
        
        inputs, labels, lengths = batch

        
        output,attention_weights = model(inputs,lengths)
        
        pred = torch.argmax(output).tolist()

        # convert to one hot
        pred_one_hot = [0,0,0]
        pred_one_hot[pred] = 1

        labels_one_hot = [0,0,0]
        labels_one_hot[labels] = 1

        # prepate JSON dict
        atten_dict = {}
        atten_dict["text"] = test_set.data[i]

        atten_dict["label"] = labels_one_hot
        atten_dict["prediction"] = pred_one_hot
        atten_dict["attention"] = attention_weights.tolist()[0]
        atten_dict["id"] = 'sample_' + str(i)
        
        samples_list.append(atten_dict)


# dump JSON for NEATVISION
with open(path,"w+") as f:
    json.dump(samples_list,f)






