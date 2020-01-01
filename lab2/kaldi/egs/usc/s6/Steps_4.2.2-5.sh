. ./path.sh
. ./cmd.sh


# Step 4.2 in folder data/local/lm_tmp we create a temporary form of LM
$KALDI_ROOT/tools/irstlm/scripts/build-lm.sh -i data/local/dict/lm_train.text -n 1 -o data/local/lm_tmp/lm_train_unigram.ilm.gz
$KALDI_ROOT/tools/irstlm/scripts/build-lm.sh -i data/local/dict/lm_train.text -n 2 -o data/local/lm_tmp/lm_train_bigram.ilm.gz


# Step 4.3 in folder data/local/nist_lm we store the comiled LM in APRA form
$KALDI_ROOT/tools/irstlm/src/compile-lm data/local/lm_tmp/lm_train_unigram.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > data/local/nist_lm/lm_train_unigram.arpa.gz
$KALDI_ROOT/tools/irstlm/src/compile-lm data/local/lm_tmp/lm_train_bigram.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > data/local/nist_lm/lm_train_bigram.arpa.gz

# Step 4.4 in folder data/lang we create the FST of the dictionary 
./utils/prepare_lang.sh data/local/dict "<oov>" data/local/lang data/lang

# Step 4.5 in folder data_test we create the FST of the Grammar for test,train,dev data and unigram and bigram model
./timit_format_data.sh