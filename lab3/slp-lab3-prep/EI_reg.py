import os
import warnings
import sys
import math

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import classification_report

import pandas as pd

import torch
from torch.utils.data import DataLoader

from config import EMB_PATH
from dataloading import SentenceDataset
from models import LSTMRegression
from training import train_dataset, eval_dataset
from utils.load_datasets import load_EI_Reg
from utils.load_embeddings import load_word_vectors
from plots import plot_loss


warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
torch.multiprocessing.set_sharing_strategy('file_system')

EMBEDDINGS = os.path.join(EMB_PATH, "glove.twitter.27B.50d.txt")


EMB_DIM = 50
MAX_SEQ_LEN = 60 
EMB_TRAINABLE = False
BATCH_SIZE = 128
EPOCHS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X,y = load_EI_Reg()
word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM)
DATASET = ''
for emotion in ['joy','anger','sadness','fear']:
    

    model_name = f'EI_{emotion}_reg'
    
    X_train = X[f'train-{emotion}']
    y_train = y[f'train-{emotion}']
    X_test = X[f'test-gold-{emotion}']
    y_test = y[f'test-gold-{emotion}']
    X_dev = X[f'dev-{emotion}']
    y_dev = y[f'dev-{emotion}']

    train_set = SentenceDataset(X_train, y_train, word2idx,MAX_SEQ_LEN)
    test_set = SentenceDataset(X_test, y_test, word2idx,MAX_SEQ_LEN)
    dev_set = SentenceDataset(X_dev, y_dev, word2idx,MAX_SEQ_LEN)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=4)
    dev_loader = DataLoader(dev_set, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=4)

    model = LSTMRegression(embeddings,attention_size = MAX_SEQ_LEN)

    checkpoint = torch.load('./checkpoints/Semeval2017A/LSTM_attention_B')['model']

    del checkpoint['linear.weight']
    del checkpoint['linear.bias']

    model.load_state_dict(checkpoint,strict= False)

    model.to(DEVICE)
    print(model)

    criterion = torch.nn.BCEWithLogitsLoss() 
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters,lr=0.001,weight_decay  = 1e-4) 
    
    n_classes = -1
    train_losses = []
    test_losses = []


    max_test_loss = math.inf
    for epoch in range(1, EPOCHS + 1):

        # train the model for one epoch
        train_dataset(epoch, train_loader, model, criterion, optimizer,n_classes)

        # evaluate the performance of the model, on both data sets
        train_loss, (y_train_gold, y_train_pred) = eval_dataset(train_loader,
                                                                model,
                                                                criterion,n_classes)

        test_loss, (y_test_gold, y_test_pred) = eval_dataset(dev_loader,
                                                                model,
                                                                criterion,n_classes)    
          # save best model
        if test_loss < max_test_loss : 
            max_test_loss = test_loss
            torch.save({
                'epoch' : epoch,
                'model' : model.state_dict(),
                'test_loss' : test_loss,
                'train_loss' : train_loss
                },f'./checkpoints/{DATASET}/{model_name}'
            )

        # plot and save, test and train loss
        print(f' Epoch {epoch}, Train loss: {train_loss:.4f}')
        print(f'           Test loss: {test_loss:.4f}')
        train_losses.append(train_loss)
        test_losses.append(test_loss)


    # load best model
    checkpoint = torch.load(f'./checkpoints/{DATASET}/{model_name}')
    model.load_state_dict(checkpoint['model'])

    # predict 
    _, (y_test_gold, y_test_pred) = eval_dataset(test_loader,model,criterion,n_classes)

    # store predictions at txt:


    # Save min loss 

    f = open(f'./reports/{DATASET}/{model_name}.tex',"w+")
    f.write(f'\nMin Test Loss: {min(test_losses):f}')
    f.close()

    # plot learning curve
    plot_loss(train_losses,test_losses,EPOCHS,DATASET,model_name)

