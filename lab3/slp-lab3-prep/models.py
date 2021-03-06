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

    def __init__(self, output_size, embeddings,method = 'mean',attention_size = 60, trainable_emb=False,tf_idf = False,
    return_weights = False):
        """

        Args:
            output_size(int): the number of classes
            embeddings(torch.Tensor):  the 2D matrix with the pretrained embeddings
            trainable_emb(bool): train (finetune) or freeze the weights
                the embedding layer
            
            method(string): Define inner embedding representation, values:
            * 'mean'  : u = mean(E), E = (e1,e2,..,en) the embeddings for each sample
            * 'pooling' : u = mean(E)||max(E)
            * 'attention' : define an attention layer on the word embeddings
        """

        super(BaselineDNN, self).__init__()

        self.tf_idf = tf_idf
        self.return_weights = return_weights
        # EX4

        # 1 - define the embedding layer
        # We define embeddings from pretrained embeddings
        
        num_embeddings = len(embeddings) 
        dim = len(embeddings[0])
        self.embeddings = nn.Embedding(num_embeddings,dim)
        self.output_size = output_size

        self.dropout = nn.Dropout(p = 0.3)
        
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
            self.attention = Attention(attention_size,dim,return_weights)
            IN_SIZE = dim
        elif method == 'mean':
            IN_SIZE = dim
        else :
            printf('Undefined method for representation calculation')
            exit(1)

        
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
        if self.tf_idf:
            tf_idf_weights = x[:,int(x.shape[1]/2):]
            x = x[:,:int(x.shape[1]/2)].long()
            embeddings = self.embeddings(x)
            embeddiings = embeddings * tf_idf_weights.unsqueeze(-1).expand_as(embeddings)   
        else :
            embeddings = self.embeddings(x)
        # BS x SEQ_LEN  --> BS x SEQ_LEN x EMB_DIM
     
        # 2 - construct a sentence representation out of the word embeddings
               
        
        # Compute mean         
        if self.method != 'attention':

            # reps :: BS x EMB_DIM
            # padded elements are mapped to 0.0 vector
            # we sum and devide with correct length for mean
            rep = torch.sum(embeddings,axis = 1)
            rep = rep / lengths.view((-1,1))


        # compute max, concatenate
        if self.method == 'pooling':

            max_vals,_ = torch.max(embeddings,dim = 1)
            rep = torch.cat((rep,max_vals),axis = 1)
        
        # apply attention layer
        elif self.method == 'attention':
            if self.return_weights:
                rep,atten_weights = self.attention(embeddings)
            else :
                rep = self.attention(embeddings)
        
        # 3 - transform the representations to new ones.
        
        rep = self.relu(self.lin1(rep))

        # 4 - project the representations to classes using a linear layer
        rep = self.dropout(rep)
        logits = self.lin2(rep)
        
        if self.output_size == 2:
            logits = logits.float()
    
        if self.return_weights :
            return logits,atten_weights  
        else :
            return logits


class BaseLSTM(nn.Module):
    def __init__(self,output_size, embeddings,hidden = 8, trainable_emb=False,method = 'mean',attention_size  = 60 ,bidirectional = False,tf_idf = False,
                return_weights = False):

        super(BaseLSTM, self).__init__()
        
        self.tf_idf = tf_idf

        embeddings = np.array(embeddings)
        num_embeddings, dim = embeddings.shape 
        
        self.embeddings = nn.Embedding(num_embeddings,dim)
        self.output_size = output_size
        
        self.lstm = nn.LSTM(dim,hidden_size = hidden,bidirectional = bidirectional)
        self.method = method 
        self.bidirectional = bidirectional
        self.return_weights = return_weights

        if bidirectional == True:
            
            hidden = hidden*2
        
        self.hidden = hidden 
            
        if method == 'pooling':
             # 3*HS , h_t_last || mean(h_t) || max(h_t)
             IN_SIZE = 3*hidden 
        elif method == 'attention':
            self.attention = Attention(attention_size,hidden,return_weights)
            IN_SIZE = hidden
        elif method == 'mean' :
            IN_SIZE = hidden
        else :
            printf('Undefined method for representation calculation')
            exit(1)


     
        if not trainable_emb:
            self.embeddings = self.embeddings.from_pretrained(torch.Tensor(embeddings), freeze = True)

        
        self.linear = nn.Linear(IN_SIZE,output_size)
    
    def forward(self,x,lengths):
       
        if self.tf_idf:
            tf_idf_weights = x[:,int(x.shape[1]/2):]
            x = x[:,:int(x.shape[1]/2)].long()
            embeddings = self.embeddings(x)
            embeddiings = embeddings * tf_idf_weights.unsqueeze(-1).expand_as(embeddings)   
        else :
            embeddings = self.embeddings(x)

        X = torch.nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True,enforce_sorted = False)

        ht,(hn,_) = self.lstm(X)
        hn = hn.squeeze()
        if self.bidirectional:
            # BS * DIR x ... --> BS  x ...  
            # concatenate the right way for bidirectional 
            hn = torch.cat((hn[0],hn[1]),dim = 1)

        ht, _ = torch.nn.utils.rnn.pad_packed_sequence(ht, batch_first=True)
        
        # ht :: BS x MAX_SEQ_LEN x HS
        # MAX_SEQ_LEN = max(lengths)
        
        if self.method == 'pooling':
            mean = torch.sum(ht,axis = 1) # BS x HS 
            mean = mean / lengths.view((-1,1)) 
            max_vals,_ = torch.max(ht,axis = 1) # BS x HS
            rep = torch.cat( (mean,max_vals) ,dim = 1) # BS x 2HS
            rep = torch.cat( (hn, rep),dim = 1) # BS x 3HS
            
        elif self.method == 'attention':
            
            if self.return_weights: 
                
                rep,atten_weights = self.attention(ht)
            
            else :
                
                rep = self.attention(ht)
       
        else :
            rep = hn
        
       
        out = self.linear(rep)
        if self.output_size == 2:
            out = out.float()
        
            
        if self.return_weights :
            return out,atten_weights  
        else :
            return out

            
        

class LSTMRegression(nn.Module):

    def __init__(self,embeddings,hidden = 8,attention_size = 60):
        super(LSTMRegression, self).__init__()
        embeddings = np.array(embeddings)
        num_embeddings, dim = embeddings.shape 
       

        self.embeddings = nn.Embedding(num_embeddings,dim)
        self.lstm = nn.LSTM(dim,hidden_size = hidden,bidirectional = True)
        hidden = hidden * 2
        self.attention = Attention(attention_size,hidden)
        self.linear = nn.Linear(hidden,1)
        self.sigmoid = nn.Sigmoid() 
    
    def forward(self,x,lengths):
        
        embeddings = self.embeddings(x)
        
        X = torch.nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True,enforce_sorted = False)

        ht,(_,_) = self.lstm(X)

        ht, _ = torch.nn.utils.rnn.pad_packed_sequence(ht, batch_first=True)
        rep = self.attention(ht)
        out = self.sigmoid(self.linear(rep))

        return out.squeeze()

class Attention(nn.Module):
    def __init__(self,attention_size,embedding_dim,return_weights = False):
        super(Attention, self).__init__()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.lin_attention = nn.Linear(embedding_dim,1)
        self.return_weights = return_weights
        # Define attention weights to be trainable
        self.att_weights = nn.Parameter(torch.FloatTensor(attention_size))

    def forward(self,embeddings):
        # Reshape to apply linear 
        
        (BS, SEQ_LEN , DIM) = embeddings.shape
        # BS x SEQ_LEN x DIM --> BS*SEQ_LEN  
        # applay a linear to each word 

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
        if self.return_weights : 
            return u,atten_weights
        return u
        
