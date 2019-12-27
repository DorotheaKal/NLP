#!/bin/bash
source ./path.sh
source ./cmd.sh

#Step 4 Question 1 

#Calculate Perplexity of dev data
echo "--------------------------------------"
echo "Calculate Perplexity of dev_unigram lm"
echo "--------------------------------------"
echo "Perplexity of dev_unigram lm" >> PP.txt
$KALDI_ROOT/tools/irstlm/src/compile-lm data/local/lm_tmp/lm_dev_unigram.ilm.gz --eval=./data/local/dict/lm_dev.text >> PP.txt


echo "--------------------------------------"
echo "Calculate Perplexity of dev_bigram lm"
echo "--------------------------------------"
echo "Perplexity of dev_bigram lm" >> PP.txt
$KALDI_ROOT/tools/irstlm/src/compile-lm data/local/lm_tmp/lm_dev_bigram.ilm.gz --eval=./data/local/dict/lm_dev.text >> PP.txt


#Calculate Perplexity of test data
echo "--------------------------------------"
echo "Calculate Perplexity of test_unigram lm"
echo "--------------------------------------"
echo "Perplexity of test_unigram lm" >> PP.txt
$KALDI_ROOT/tools/irstlm/src/compile-lm data/local/lm_tmp/lm_test_unigram.ilm.gz --eval=./data/local/dict/lm_test.text >> PP.txt


echo "--------------------------------------"
echo "Calculate Perplexity of test_bigram lm"
echo "--------------------------------------"
echo "Perplexity of test_bigram lm" >> PP.txt
$KALDI_ROOT/tools/irstlm/src/compile-lm data/local/lm_tmp/lm_test_bigram.ilm.gz --eval=./data/local/dict/lm_test.text >> PP.txt
