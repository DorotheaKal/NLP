
source ./data/path.sh
$KALDI_ROOT/tools/irstlm/scripts/build-lm.sh -i ./dict/lm_train.text -n 1 -o ./lm_tmp/unigram.ilm.gz
$KALDI_ROOT/tools/irstlm/scripts/build-lm.sh -i ./dict/lm_train.text -n 2 -o ./lm_tmp/bigram.ilm.gz
$KALDI_ROOT/tools/irstlm/src/compile-lm ./lm_tmp/bigram.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > ./nist_lm/bigram.arpa.gz
$KALDI_ROOT/tools/irstlm/src/compile-lm ./lm_tmp/unigram.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > ./nist_lm/unigram.arpa.gz

./utils/prepare_lang.sh ./data/local/dict 'oov' ./data/local/lang ./data/lang/ 
pwd
#./local/format_data.sh > ./lang/G.fst 