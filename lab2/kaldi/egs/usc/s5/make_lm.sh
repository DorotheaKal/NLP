source ./cmd.sh
source ./path.sh

$KALDI_ROOT/tools/irstlm/scripts/build-lm.sh -i ./data/local/dict/lm_train.text -n 1 -o ./data/local/lm_tmp/unigram_train.ilm.gz
$KALDI_ROOT/tools/irstlm/scripts/build-lm.sh -i ./data/local/dict/lm_train.text -n 2 -o ./data/local/lm_tmp/bigram_train.ilm.gz
$KALDI_ROOT/tools/irstlm/src/compile-lm ./data/local/lm_tmp/bigram_train.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > ./data/local/nist_lm/bigram_train.arpa.gz
$KALDI_ROOT/tools/irstlm/src/compile-lm ./data/local/lm_tmp/unigram_train.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > ./data/local/nist_lm/unigram_train.arpa.gz

./../../timit/s5/utils/prepare_lang.sh  ./data/local/dict "<oov>" ./data/local/lm_tmp ./data/lang
#./local/format_data.sh > ./lang/G.fst 
