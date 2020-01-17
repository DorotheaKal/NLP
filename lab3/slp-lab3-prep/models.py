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
            self.tanh = nn.Tanh()
            self.softmax = nn.Softmax()
            self.lin_attention = nn.Linear(dim,1)
            # Define attention weights to be trainable
            self.att_weights = torch.Parameter(torch.FloatTensor(attention_size))
            IN_SIZE = dim
        else :
            IN_SIZE = dim
        
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
               
       
        
        # Compute mean         
        if self.method != 'attention':

            # representations :: BS x EMB_DIM
            # OOV words are mapped to 0 vector
            # we sum and devide with correct length for mean
            representations = torch.sum(embeddings,axis = 1)
            representations = representations / lengths.view((-1,1))


        # compute max, concatenate
        if self.method == 'pooling':

            max_vals,_ = torch.max(embeddings,dim = 1)
            representations = torch.cat((representations,max_vals),axis = 1)
        
        # apply attention layet
        elif self.method == 'attention':
            # Reshape to apply linear 
            (BS, SEQ_LEN ,_) = embeddings.shape
            # BS x SEQ_LEN x DIM --> BS*SEQ_LEN  
            # applay a linear to each word 
            representations  = self.lin_attention(embeddings.view((-1,self.dim)))
            # reshape to  BS x SEQ_LEN
            representations = representations.view((BS,SEQ_LEN))
            # non-linearity
            representations = self.tanh(representations)
            # SEQ_LEN x 1 
            atten_weights = self.softmax(representations)
            # BS x SEQ_LEN x DIM  @  SEQ_LEN x 1 -->  BS x DIM 
            representations = torch.matmul(embeddings,atten_weights)
            import ipdb; ipdb.set_trace()
        # 3 - transform the representations to new ones.
        representations = self.relu(self.lin1(representations))

        # 4 - project the representations to classes using a linear layer
        
        logits = self.lin2(representations)
        
        if self.output_size == 2:
            logits = logits.view((-1)).float()
        
        return logits
