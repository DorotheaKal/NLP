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

    def __init__(self, output_size, embeddings,method = 'mean',attention_size = 60, trainable_emb=False):
        """

        Args:
            output_size(int): the number of classes
            embeddings(bool):  the 2D matrix with the pretrained embeddings
            trainable_emb(bool): train (finetune) or freeze the weights
                the embedding layer
            
            method(string): Define inner embedding representation, values:
            * 'mean'  : u = mean(E), E = (e1,e2,..,en) the embeddings for each sample
            * 'pooling' : u = mean(E)||max(E)
            * 'attention' : define an attention layer on the word embeddings
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
    
        self.method = method 
        if method == 'pooling':
             IN_SIZE = 2*dim 
        elif method == 'attention':
            self.attention = Attention(attention_size,dim)
            IN_SIZE = dim
        else :
            IN_SIZE = dim
        
        self.lin1 = nn.Linear(IN_SIZE,LATENT_SIZE)
        self.relu = nn.ReLU()
        
        
        # 5 - define the final Linear layer which maps
        # the representations to the classes
        

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
               
       
        
        # Compute mean         
        if self.method != 'attention':

            # reps :: BS x EMB_DIM
            # OOV words are mapped to 0 vector
            # we sum and devide with correct length for mean
            rep = torch.sum(embeddings,axis = 1)
            rep = rep / lengths.view((-1,1))


        # compute max, concatenate
        if self.method == 'pooling':

            max_vals,_ = torch.max(embeddings,dim = 1)
            rep = torch.cat((rep,max_vals),axis = 1)
        
        # apply attention layer
        elif self.method == 'attention':
            rep = self.attention(embeddings)
        
        # 3 - transform the representations to new ones.
        rep = self.relu(self.lin1(rep))

        # 4 - project the representations to classes using a linear layer
        
        logits = self.lin2(rep)
        
        if self.output_size == 2:
            logits = logits.view((-1)).float()
        
        return logits

class BaseLSTM(nn.Module):
    def __init__(self,output_size, embeddings,hidden = 8, trainable_emb=False,method = 'mean',attention_size = 60,bidirectional = False):

        super(BaseLSTM, self).__init__()
        
        embeddings = np.array(embeddings)
        num_embeddings, dim = embeddings.shape 
        
        self.embeddings = nn.Embedding(num_embeddings,dim)
        self.output_size = output_size
        
        self.lstm = nn.LSTM(dim,hidden_size = hidden)
        self.method = method 
        self.bidirectional = bidirectional
      

        if bidirectional == True:
            self.lstm_rev = nn.LSTM(dim,hidden_size = hidden)
            hidden = hidden*2
        self.hidden = hidden 
            
        if method == 'pooling':
             # 3*HS , h_t_last || mean(h_t) || max(h_t)
             IN_SIZE = 3*hidden 
        elif method == 'attention':
            self.attention = Attention(attention_size,hidden)
            IN_SIZE = hidden
        else :
            IN_SIZE = hidden


     
        if not trainable_emb:
            self.embeddings = self.embeddings.from_pretrained(torch.Tensor(embeddings), freeze = True)

        
        self.linear = nn.Linear(IN_SIZE,output_size)
    
    def forward(self,x,lengths):
        embeddings = self.embeddings(x)
        # :: BS x SEQ_LEN x EMB_DIM 
        
        BS,SEQ_LEN  = x.shape
    
        # LSTM requires input SEQ_LEN x BS x EMB_DIM
        ht,_ = self.lstm(torch.transpose(embeddings, 0, 1))
        
        if self.bidirectional:
        
            x_rev = torch.Tensor(np.flip(x.numpy(),1).copy())
            for i in range(BS):
                
                x_rev[i] = torch.cat((x_rev[i,SEQ_LEN - lengths[i]:],x_rev[i,:SEQ_LEN - lengths[i]]),dim = 0)
            
            x_rev = x_rev.long()
            embeddings_rev = self.embeddings(x_rev)
            ht_rev,_ = self.lstm_rev(torch.transpose(embeddings_rev,0 , 1 ))
        
        # ht :: SEQ_LEN x BS x HS
        
        ## Need to index by lengths, drop unwanted hidden states...
        
        # ht_last :: BS x HS
        
        ht_last = torch.zeros((BS,self.hidden))
        
        for i in range(BS):
            if (lengths[i] > ht.shape[0]):
                lengths[i] = ht.shape[0]
                continue

            ht[lengths[i]:,i,:] = 0 # zero unwanted states
            

            
            if self.bidirectional:
                ht_rev[lengths[i]:,i,:] = 0
                
                ht_last[i] = torch.cat((ht[lengths[i]-1,i,:],ht_rev[lengths[i]-1,i,:]))
            else :
                ht_last[i] = ht[lengths[i]-1,i,:] # save last hidden       
        
        if self.bidirectional :
    
            ht = torch.cat((ht,ht_rev),dim = 2)

        ht =  torch.transpose(ht,0,1) # transpose back to previous shape
        if self.method == 'pooling':
            mean = torch.sum(ht,axis = 1) # BS x HS 
            mean = mean / lengths.view((-1,1)) 
            max_vals,_ = torch.max(ht,axis = 1) # BS x HS
            rep = torch.cat( (mean,max_vals) ,dim = 1) # BS x 2HS
            rep = torch.cat( (ht_last, rep),dim = 1) # BS x 3HS
            
        elif self.method == 'attention':
            rep = self.attention(ht)
        else :
            rep = ht_last
        
        out = self.linear(rep)
        if self.output_size == 2:
            out = out.view((-1)).float()
        
        return out

class Attention(nn.Module):
    def __init__(self,attention_size,embedding_dim):
        super(Attention, self).__init__()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.lin_attention = nn.Linear(embedding_dim,1)
        # Define attention weights to be trainable
        self.att_weights = nn.Parameter(torch.FloatTensor(attention_size))
    
    def forward(self,embeddings):
        # Reshape to apply linear 
        
        (BS, SEQ_LEN , DIM) = embeddings.shape
        # BS x SEQ_LEN x DIM --> BS*SEQ_LEN  
        # applay a linear to each word 
        #import ipdb; ipdb.set_trace()
        u  = self.lin_attention(embeddings.reshape((-1,DIM)))
        # reshape to  BS x SEQ_LEN
        u = u.reshape((BS,SEQ_LEN))
        # non-linearity
        u = self.tanh(u)
        # SEQ_LEN x 1 
        atten_weights = self.softmax(u)
        
        # Expand attention_weights to embeddings size and multiply 
        # BS x SEQ_LEN x DIM
        u = torch.mul(embeddings,atten_weights.unsqueeze(-1).expand_as(embeddings))
        # BS x DIM
        u = torch.sum(u,dim = 1)
        return u
        
