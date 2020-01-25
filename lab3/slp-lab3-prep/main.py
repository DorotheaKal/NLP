import os
import warnings
import sys
import math

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

import pandas as pd


import torch
from torch.utils.data import DataLoader

from config import EMB_PATH
from dataloading import SentenceDataset
from models import BaselineDNN,BaseLSTM
from training import train_dataset, eval_dataset
from utils.load_datasets import load_MR, load_Semeval2017A
from utils.load_embeddings import load_word_vectors
from plots import plot_loss


warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
torch.multiprocessing.set_sharing_strategy('file_system')
########################################################
# Configuration
########################################################


# Download the embeddings of your choice
# for example http://nlp.stanford.edu/data/glove.6B.zip

# 1 - point to the pretrained embeddings file (must be in /embeddings folder)
EMBEDDINGS = os.path.join(EMB_PATH, "glove.6B.50d.txt")

# 2 - set the correct dimensionality of the embeddings
EMB_DIM = 50
MAX_SEQ_LEN = 60 
EMB_TRAINABLE = False
BATCH_SIZE = 128
EPOCHS = 2
DATASET =  "Semeval2017A"  # options: "MR", "Semeval2017A"
TF_IDF = False
# if your computer has a CUDA compatible gpu use it, otherwise use the cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#######################################################
# Define PyTorch datasets and dataloaders
########################################################

# load word embeddings
print("loading word embeddings...")
word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM)

# load the raw data
if DATASET == "Semeval2017A":
    X_train, y_train, X_test, y_test = load_Semeval2017A()
elif DATASET == "MR":
    X_train, y_train, X_test, y_test = load_MR()
else:
    raise ValueError("Invalid dataset")

# convert data labels from strings to integers


le = LabelEncoder()
le.fit(list(set(y_train)))
y_train = le.transform(y_train)
y_test = le.transform(y_test)

n_classes = len(list(le.classes_))
# EX 1
print('\n\033[1mQuestion 1:\033[0m\n')
print('\nLabels for 10 first training examples:\n')
print(f'Original: {le.inverse_transform(y_train[:10])}\n')
print(f'After LabelEncoder: {y_train[:10]}\n')

# Define our PyTorc-based Dataset

try :
    DEBUG = True if sys.argv[2] == 'debug' else False 
except: 
    DEBUG = False
if DEBUG:
    # if debug only process one batch 
    train_set = SentenceDataset(X_train[:BATCH_SIZE], y_train[:BATCH_SIZE], word2idx,MAX_SEQ_LEN,tf_idf = TF_IDF)
    test_set = SentenceDataset(X_test[:BATCH_SIZE], y_test[:BATCH_SIZE], word2idx,MAX_SEQ_LEN,tf_idf = TF_IDF)
else:
    train_set = SentenceDataset(X_train, y_train, word2idx,MAX_SEQ_LEN,tf_idf = TF_IDF)
    test_set = SentenceDataset(X_test, y_test, word2idx,MAX_SEQ_LEN,tf_idf = TF_IDF)

print(idx2word[1])

print('\n\033[1mQuestion 2:\033[0m\n')
for i in range(10):
    print('Tokenized sample:')
    print(train_set.data[i])
    print()
      
print('\n\033[1mQuestion 3:\033[0m\n')
for i in range(15,20):
    print('Original sample:')
    print(X_train[i])
    print()
    print('Transformed sample:')
    print(train_set[i])
    print()


# EX4 - Define our PyTorch-based DataLoader

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=4)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=4)



#############################################################################
# Model Definition (Model, Loss Function, Optimizer)
#############################################################################

'''
Define a model as a parameter, syntax:
[DNN | LSTM]_[mean | attention | pooling]_B?
'''

try :
    model_name = sys.argv[1]
except :
    print('Argument Required, model name')
    exit(1)

args = model_name.split('_')
if args[0] == 'DNN' :

    model = BaselineDNN(output_size=n_classes,  # EX8
                    embeddings=embeddings,
                    method = args[1],
                    attention_size=MAX_SEQ_LEN,
                    trainable_emb=EMB_TRAINABLE,tf_idf = TF_IDF)
  
elif args[0] == 'LSTM':
    
    bidirectional = True if args == 3 and args[2] == 'B' else False

    model = BaseLSTM(output_size=n_classes,  
                    embeddings=embeddings,
                    method = args[1],
                    attention_size=MAX_SEQ_LEN,
                    bidirectional= bidirectional,
                    trainable_emb=EMB_TRAINABLE,tf_idf = TF_IDF)
else :
    print('Invalid model name')
    exit(1)



# move the mode weight to cpu or gpu
model.to(DEVICE)
print(model)

# We optimize ONLY those parameters that are trainable (p.requires_grad==True)
criterion = torch.nn.BCEWithLogitsLoss() if n_classes == 2 else torch.nn.CrossEntropyLoss()  # EX8 
parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(parameters,lr=0.001,weight_decay  = 1e-4)  # EX8

train_losses = []
test_losses = []

#############################################################################
# Training Pipeline
#############################################################################




max_test_loss = math.inf
for epoch in range(1, EPOCHS + 1):

    # train the model for one epoch
    train_dataset(epoch, train_loader, model, criterion, optimizer,n_classes)

    # evaluate the performance of the model, on both data sets
    train_loss, (y_train_gold, y_train_pred) = eval_dataset(train_loader,
                                                            model,
                                                            criterion,n_classes)

    test_loss, (y_test_gold, y_test_pred) = eval_dataset(test_loader,
                                                         model,
                                                         criterion,n_classes)    
    
    print(f' Epoch {epoch}, Train loss: {train_loss:.4f}')
    print(f'           Test loss: {test_loss:.4f}')
    train_losses.append(train_loss)
    test_losses.append(test_loss)

    if test_loss < max_test_loss : 
        max_test_loss = test_loss
        torch.save({
            'epoch' : epoch,
            'model' : model.state_dict(),
            'optimizer' : optimizer.state_dict(), 
            'test_loss' : test_loss,
            'train_loss' : train_loss
            },f'./checkpoints/{DATASET}/{model_name}'
        )

# load best model
checkpoint = torch.load(f'./checkpoints/{DATASET}/{model_name}')
model.load_state_dict(checkpoint['model'])
_, (y_test_gold, y_test_pred) = eval_dataset(test_loader,model,criterion,n_classes)

print('\n\033[1mQuestion 10, Classification Report:\033[0m\n')
print(classification_report(y_test_gold,y_test_pred))
report = classification_report(y_test_gold,y_test_pred,output_dict=True)
df = pd.DataFrame(report).transpose()
f = open(f'./reports/{DATASET}/{model_name}.tex',"w+")
f.write(df.to_latex())
f.write(f'\nMin Test Loss: {min(test_losses):f}')
f.close()


print('\n\033[1mQuestion 10, Plot:\033[0m\n')
plot_loss(train_losses,test_losses,EPOCHS,DATASET,model_name)


