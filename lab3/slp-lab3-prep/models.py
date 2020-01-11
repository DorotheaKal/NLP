import torch

from torch import nn
import numpy as np
import torch.nn.functional as F

class BaselineDNN(nn.Module):
    """
    1. We embed the words in the input texts using an embedding layer
    2. We compute the min, mean, max of the word embeddings in each sample
       and use it as the feature representation of the sequence.
    4. We project with a linear layer the representation
       to the number of classes.ngth)
    """

    def __init__(self, output_size, embeddings, trainable_emb=False):
        """

        Args:
            output_size(int): the number of classes
            embeddings(bool):  the 2D matrix with the pretrained embeddings
            trainable_emb(bool): train (finetune) or freeze the weights
                the embedding layer
        """

        super(BaselineDNN, self).__init__()

        # EX4

        # 1 - define the embedding layer
        # We define embeddings from pretrained embeddings
        num_embeddings = len(embeddings)
        dim = len(embeddings[0])
        self.dim = dim
        self.embeddings = nn.Embedding(num_embeddings,dim)
        
        # 2 - initialize the weights of our Embedding layer
        # from the pretrained word embeddings
        # 3 - define if the embedding layer will be frozen or finetuned

        if not trainable_emb:
            # Load from pretrained and freeze..
            self.embeddings = self.embeddings.from_pretrained(torch.Tensor(embeddings), freeze = True)
        


        # EX5     
        # 4 - define a non-linear transformation of the representations
    
        
        self.lin1 = nn.Linear(dim,int(dim/2))
        self.relu = nn.ReLU()
        
        
        # 5 - define the final Linear layer which maps
        # the representations to the classes
        
        if output_size == 2:
            self.sig = nn.Sigmoid()
            self.lin2 = nn.Linear(int(dim/2),1)
        else:
            self.lin2 = nn.Linear(int(dim/2),output_size)
        self.output_size = output_size
    def forward(self, x, lengths):
        """
        This is the heart of the model.
        This function, defines how the data passes through the network.

        Returns: the logits for each class

        """
        # EX6
        # 1 - embed the words, using the embedding layer
        # x ==  BS x MAX_LEN
        
        embeddings = self.embeddings(x)
        # BS x MAX_LEN  -> BS x MAX_LEN x EMB_DIM 
        # 2 - construct a sentence representation out of the word embeddings
        batch_size = embeddings.shape[0]
        representations = torch.zeros((batch_size,self.dim))
        # BS x EMB_DIM

        for i in range(batch_size):
            representations[i,:] = torch.mean(embeddings[i,:lengths[i],:],axis = 0)

        #import ipdb;ipdb.set_trace()


        # 3 - transform the representations to new ones.
        representations = self.relu(self.lin1(representations))

        # 4 - project the representations to classes using a linear layer
        logits = self.lin2(representations)
        if self.output_size == 2:
            logits = self.sig(logits)
            logits = logits.view((-1))
            
            

        return logits
