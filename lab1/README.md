In this lab we experiment with a simple orthographer built on Finite State Tansducers (FSTs) with the OpenFST library. For the language model we constuct a corpus of publicly available books. We decide that the best approach is one based on word-level representations, as Unigram or Bigram models reduce accuracy. On the small dataset we experimented on we achieved an accuracy of **0.618**.

On the second part we experimented with different word embeddings on the classification tsk on classical IMDB review dataset:
* Locally trained word2vec 
* GoogleNews embeddings
* tf-idf weighted embeddings

We also experimented with common classifiers:
* Logistic Regression
* k-NN
* SVM 
