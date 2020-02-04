# NLP
This repository containts code for the lab exercises of the NLP cource @ NTUA. 

The course introduces traditioanal natural language processing tools and concepts such as Finite State Transducers, Hidden Markov Models , N-grams, embeddings, frequency domain approach to speech synthesis and recognition. The modern NLP DNN baseline using LSTMs and an Attention layer is also examined.


Authored by [nikitas-theo](https://github.com/nikitas-theo) and [dorotheaKal](https://github.com/DorotheaKal) 
 
## Lab 1 : Introduction to Language Representations

 We experimented with a weighted minimun distance orthographer built on Finite State Tansducers (FSTs) with the [OpenFST](http://www.openfst.org/twiki/bin/view/FST/WebHome) library. The language model was trained on a corpus of publicly available books. We decided that the best approach is one based on word-level representations, as Unigram or Bigram models reduce accuracy. On the small dataset we experimented on we achieved an accuracy of 0.618.

We visualize the proposed corrected words for the input word *cit*.

<img src="./lab1/img/min_cit.jpg" width="400" height="auto" >


 For the second part we worked on the classical IMDB review dataset for emotion classification. Different semantic representations were used:
* Locally trained word2vec embeddings
* GoogleNews embeddings
* tf-idf weighted embeddings


## Lab 2 : Speech Recognition with the Kaldi Toolkit

A word / phoneme recognition system is implemented with [Kaldi](https://kaldi-asr.org/), a framework used for state of the art speech applications. 

The features used were the Mel-Frequency Cepstral Coefficients (MFCCs) from 4 speakers (2 male, 2 female), on the USC-TIMIT dataset. The MFCCs are computed as a tansformation of the STFT.  The Mel Filterbank, desinged to model the logarithmic nature of human sound perception,  is applied and a final DCT transfrom ensures feature independence. More on the MFCCs on [this](https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html) blog.


For the a priori probabilities we used language models, trained on transcription information. The accoustic model is a triphone-based HMM, to include speech context information. The final estimation is Bayesian formulated. The prior probability for a word W, is given by the language model. The likelihood P(X|W) is calculated from the accoustic model.

<img src="./lab2/img/bayesian.png" width="600" height="auto">


The diagram below corrseponds to the typical ASR (Automatic Speech Recognition) system we implemented. 

<img src="./lab2/latex_source/ASR_diagram.png" width="500" height="auto">

## Lab 3 : Sentiment Classification with DNNs

We implemented Deep Neural Network models for text processing and classification. The PyTorch framework was used.
The task is emotion recognition. [GloVe](https://nlp.stanford.edu/projects/glove/) Twitter word embeddings (27B tokens, 50d) were used to exploit the emotion sensitive information in tweets e.g. emojis. 

We used 2 datasets:
* Sentence Polarity Dataset [Pang and Lee, 2005] containing 5331 positive
and 5331 negative movie reviews from Rotten Tomatoes, for binary-classification
(positive, negative).

* Semeval 2017 Task4-A [Rosenthal et al.,2017]. This dataset contains tweets representing 3 classes (positive, negative,neutral) with 49570
training samples and 12284 test (validation) samples.

We experimented with a variety of models, with 2 baseline architectures, a DNN and an LSTM. We used different intermediate representations based on word embeddings, such as max and mean pooling and tf-idf weights. We also implemented an Attention layer on both models. The accuracy for our models is close to the ~65% value.


We visualize the attention layer weights of our LSTM models on insightful samples from the test set using [NeAt (Neural Attention) Vision](https://github.com/cbaziotis/neat-vision). We conclude that the model has succesfully focused on emotional words that for the most part contribute to it's classification accuracy.  


* Our model correctly predicts  the **positive** label
<img src="./lab3/latex_source/473_LSTM.png" width="550" height="auto">


* Our model incorrectly labels as **negative** the **neutral** tweet
<img src="./lab3/latex_source/500_LSTM.png" width="650" height="auto">



* Our model correctly predicts the **positive** label
<img src="./lab3/latex_source/480.png" width="1400" height="auto">



Finally we applied Transfer Learning, utilizing pre-trained weights on the [SemEval-2017 Task4-A](http://alt.qcri.org/semeval2017/task4/) with target dataset the [SemEval-2018 Task1](https://competitions.codalab.org/competitions/17751), affect in tweets. We focused on the Emotion Intensity task (EI Regression). For 4 different emotions (joy,anger,sadness,fear) we predict the intesity of a specific emotion presented in the tweet. Our implementaion is based on [Baziotis et al., 2018]. The approach gave good results and we showcase the Learning curve for **joy** after training for 50 epochs. 


<img src="./lab3/latex_source/img/EI_Reg/EI_joy_reg_loss.png" width="500" height="auto">







