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

    def __init__(self, output_size, embeddings,pooling = False, trainable_emb=False):
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
        self.embeddings = nn.Embedding(num_embeddings,dim)
        self.output_size = output_size
        
        # 2 - initialize the weights of our Embedding layer
        # from the pretrained word embeddings

        # 3 - define if the embedding layer will be frozen or finetuned

        if not trainable_emb:
            # Load from pretrained and (freeze = True <=> requires_grad = False).
            self.embeddings = self.embeddings.from_pretrained(torch.Tensor(embeddings), freeze = True)
        


        # EX5     
        # 4 - define a non-linear transformation of the representations
    
        LATENT_SIZE = 1024 
        
        self.pooling = pooling 
        
        IN_SIZE = 2*dim if pooling else dim 
        
        self.lin1 = nn.Linear(IN_SIZE,LATENT_SIZE)
        self.relu = nn.ReLU()
        
        
        # 5 - define the final Linear layer which maps
        # the representations to the classes
        
        # For bin classficiation output dim is 1
        if output_size == 2 :
            output_size = 1
    
        self.lin2 = nn.Linear(LATENT_SIZE,output_size)

    
    def forward(self, x, lengths):
        """
        Returns: the logits for each class
        """
        # EX6
        # 1 - embed the words, using the embedding layer
        # x ::  BS x SEQ_LEN
        
        embeddings = self.embeddings(x)
        # BS x SEQ_LEN  --> BS x SEQ_LEN x EMB_DIM
     
        # 2 - construct a sentence representation out of the word embeddings
               
        # representations :: BS x EMB_DIM
        # OOV words are mapped to 0 vector
        # we sum and devide with correct length for mean
        
        representations = torch.sum(embeddings,axis = 1)
        representations = representations / lengths.view((-1,1))
        
        if self.pooling:
            max_vals,_ = torch.max(embeddings,dim = 1)
            representations = torch.cat((representations,max_vals),axis = 1)
        
        # 3 - transform the representations to new ones.
        representations = self.relu(self.lin1(representations))

        # 4 - project the representations to classes using a linear layer
        
        logits = self.lin2(representations)
        
        if self.output_size == 2:
            logits = logits.view((-1)).float()
        
        return logits


class BaseLSTM(nn.Module):
    def __init__(self,output_size, embeddings,hidden = 8, trainable_emb=False):

        super(BaseLSTM, self).__init__()
        
        embeddings = np.array(embeddings)
        num_embeddings, dim = embeddings.shape 
        
        self.embeddings = nn.Embedding(num_embeddings,dim)
        self.output_size = output_size
        
        self.lstm = nn.LSTM(dim,hidden_size = hidden)
        if not trainable_emb:
            self.embeddings = self.embeddings.from_pretrained(torch.Tensor(embeddings), freeze = True)

        if output_size == 2 :
            output_size = 1
    
        # 3*HS , h_t_last || mean(h_t) || max(h_t)
        self.linear = nn.Linear(3*hidden,output_size)
    
    def forward(self,x,lengths):
        embeddings = self.embeddings(x) # :: BS x SEQ_LEN x EMB_DIM 
        
        # LSTM requires input SEQ_LEN x BS x EMB_DIM
        ht,_, = self.lstm(torch.transpose(embeddings, 0, 1))
        # ht :: SEQ_LEN x BS x HS
        
        ## Need to index by lengths, drop unwanted hidden states...
        
        # ht_last :: BS x HS
        ht_last = torch.zeros((ht.shape[1],ht.shape[2]))
        
        for i in range(ht.shape[1]):
            if (lengths[i] > ht.shape[0]):
                lengths[i] = ht.shape[0]

            ht[lengths[i]:,i,:] = 0 # zero unwanted states
            ht_last[i] = ht[lengths[i]-1,i,:] # save last hidden

        ht =  torch.transpose(ht,0,1) # transpose back to previous shape
       
        mean = torch.sum(ht,axis = 1) # BS x HS 
        mean = mean / lengths.view((-1,1)) 
        max_vals,_ = torch.max(ht,axis = 1) # BS x HS
        rep = torch.cat( (mean,max_vals) ,dim = 1) # BS x 2HS
        rep = torch.cat( (ht_last, rep),dim = 1) # BS x 3HS
        out = self.linear(rep)
        
        if self.output_size == 2:
            out = out.view((-1)).float()
        
        return out
