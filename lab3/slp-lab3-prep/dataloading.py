from torch.utils.data import Dataset
import nltk
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer


class SentenceDataset(Dataset):
    """
    Our custom PyTorch Dataset, for preparing strings of text (sentences)
    What we have to do is to implement the 2 abstract methods:

        - __len__(self): in order to let the DataLoader know the size
            of our dataset and to perform batching, shuffling and so on...

        - __getitem__(self, index): we have to return the properly
            processed data-item from our dataset with a given index
    """

    def __init__(self, X, y, word2idx,MAX_SEQ_LEN,tf_idf = False):
        """
        In the initialization of the dataset we will have to assign the
        input values to the corresponding class attributes
        and preprocess the text samples

        -Store all meaningful arguments to the constructor here for debugging
         and for usage in other methods
        -Do most of the heavy-lifting like preprocessing the dataset here


        Args:
            X (list): List of training samples
            y (list): List of training labels
            word2idx (dict): a dictionary which maps words to indexes
            MAX_SEQ_LEN: maximum length of sequence
            tf_idf : calculate tf-idf weights over data
        """
        
        # Download pre-trained tokenizer if necessary 
        try:
            nltk.data.find('tokenizers/punkt/english.pickle')
        except LookupError:
            nltk.download('punkt/english.pickle')
        
        # tokenize samples
        # lower() needed for word2idx 
        self.data = [ list(map(lambda x:x.lower(),nltk.word_tokenize(x))) for x in X]
        self.tf_idf = tf_idf

        if tf_idf:
            vocab = word2idx.copy()
            # hack for CountVectorizer
            vocab['dummy-word'] = 0
            data_counts = CountVectorizer(vocabulary = vocab,tokenizer = lambda x : x, preprocessor=None, lowercase=False).fit_transform(self.data)
            self.tf_idf_data =  TfidfTransformer().fit_transform(data_counts)       
        




        
        self.labels = y
        
        self.word2idx = word2idx

        # We arbirtarily choose max sentence length
        self.max_sent_length = MAX_SEQ_LEN
        

    def __len__(self):
        """
        Must return the length of the dataset, so the dataloader can know
        how to split it into batches

        Returns:
            (int): the length of the dataset
        """

        return len(self.data)

    def __getitem__(self, index):
        """
        Returns the _transformed_ item from the dataset

        Args:
            index (int):

        Returns:
            (tuple):
                * example (ndarray): vector representation of a training example
                * label (int): the class label
                * length (int): the length (tokens) of the sentence

        Examples:
            For an `index` where:
            ::
                self.data[index] = ['this', 'is', 'really', 'simple']
                self.target[index] = "neutral"

            the function will have to return something like:
            ::
                example = [  533  3908  1387   649   0     0     0     0]
                label = 1
                length = 4
        """

        # EX3

        sentence = self.data[index]
        label = self.labels[index]
        length = len(sentence)
        
        example = []
        for token in sentence:
            # get indexes
            if token in self.word2idx.keys():
                example.append(self.word2idx[token])
            else :
                example.append(self.word2idx['<unk>']) 

        if length >= self.max_sent_length:
            # reduce size if necessary 
            example = example[:self.max_sent_length]
            length = self.max_sent_length
        else :
            # else pad sequence
            example = example + [0]*(self.max_sent_length-length)
            
        example = np.array(example)
        
        if self.tf_idf: 

            # Get tf_idf weights 
            # Get non zero indices from sparce matrix


            # get corresponding scores
            tf_idf_weights = [self.tf_idf_data[index, x] for x in example[:length] ]
            # pad accordingly
            if length >= self.max_sent_length:
                tf_idf_weights = tf_idf_weights[:self.max_sent_length] 
            else :
                tf_idf_weights = tf_idf_weights + [0]*(self.max_sent_length-length)

            tf_idf_weights = np.array(tf_idf_weights)
            example = np.concatenate((example,tf_idf_weights))
            

        
        return example, label, length

