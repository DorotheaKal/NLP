# NLP
Thie repository containts code for the lab exercises of the NLP cource @ NTUA. 

The course introduces traditional natural language processing tools and concepts such as Finite State Transducers, Hidden Markov Models , N-grams, embeddings, signal based approach to speech recognition and synthesis.  

Authored by [nikitas-theo]() and [dorotheaKal]() 

**Lab 1**

 We experiment with a weighted minimun distance orthographer built on Finite State Tansducers (FSTs) with the OpenFST library. For the language model we constuct a corpus of publicly available books. We decide that the best approach is one based on word-level representations, as Unigram or Bigram models reduce accuracy. On the small dataset we experimented on we achieved an accuracy of **0.618**.

 For the second part we worked on the  classical IMDB review dataset for classification. Different semantic representations (word embeddings) were used:
* Locally trained word2vec 
* GoogleNews embeddings
* tf-idf weighted embeddings

<img src="./lab1/img/min_cit.jpg" width="400" height="auto">


**Lab 2**

A word/phoneme recognition system is implemented with [Kaldi](https://github.com/kaldi-asr/kaldAi), a framework used for state of the art speech aplications. 

For features we used the Mel-Frequency Cepstral Coefficients (MFCCs) from 4 speakers (2 male, 2 female), on the USC-TIMIT dataset. 

The MFCCs are computed as a tansformation of the STFT after he Mel Filterbank, desinged to model the logarithmic nature of human sound perception,  is applied. A DCT transfrom ensures feature independence. 

For the a priori probabilities we used language models, trained on transcription information. The accoustic model is a triphone-based HMM, to include speech context information. The final estimation is Bayesian formulated:

$$ \hat{W} = arg max_W P(W|X) = arg max_W \frac{P(X|W)P(W)}{P(X)} = arg max_W P(X|W)P(W) $$  


<img src="./lab2/latex_source/ASR_diagram.png" width="500" height="auto">

**Lab 3**

We implement Deep Neural Network models for text processing and classification. The PyTorch framework was used.
The task is emotion recognition. GloVe Twitter word embeddings (27B tokens, 50d) were used to exploit the emotion sensitive information in tweets e.g. emojis. 

We used 2 datasets:
* Sentence Polarity Dataset [Pang and Lee, 2005] containing 5331 positive
and 5331 negative movie reviews from Rotten Tomatoes, for binary-classification
(positive, negative).

* Semeval 2017 Task4-A [Rosenthal et al.,2017]. This dataset contains tweets representing
3 classes (positive, negative,neutral) with 49570
training samples and 12284 test (validation) samples.

We experimented with a variety of model  with 2 baseline architectures, a DNN and an LSTM approach. We used different intermediate representations based on word embeddings, such as max and mean pooling and tf-idf. We also implemented an Attention layer on both models.

We visualize the attention layer results of our LSTM model on insightful samples from the test set:


* Our model correctly predicts **positive** label
<img src="./lab3/latex_source/473_LSTM.png" width="600" height="auto">


* Our model incorrectly labels as negative the **neutral** tweet.

<img src="./lab3/latex_source/500_LSTM.png" width="500" height="auto">



* Correctrly **positive** labeled
<img src="./lab3/latex_source/480.png" width="800" height="auto">



Finally we applied Transfer Learning from the SemEval-2017 Task4-A with target dataset the SemEval-2018 Task1, affect in tweets. We focused on the Emotion Intensity task (EI Regression) for 4 different emotions (joy,anger,sadness,fear). Our implementaion is based on [Baziotis et al., 2018]. The approach gave good results and we showcase the Learning curve for **joy**:



<img src="./lab3/latex_source/img/EI_Reg/EI_joy_reg_loss.png" width="500" height="auto">







