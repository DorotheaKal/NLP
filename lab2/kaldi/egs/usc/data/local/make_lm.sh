export IRSTLM=$HOME/kaldi/tools/irstlm
~/kaldi/tools/irstlm/scripts/build-lm.sh -i ./dict/lm_train.text -n 1 -o ./lm_tmp/unigram.ilm.gz
~/kaldi/tools/irstlm/scripts/build-lm.sh -i ./dict/lm_train.text -n 2 -o ./lm_tmp/bigram.ilm.gz
~/kaldi/tools/irstlm/src/compile-lm ./lm_tmp/bigram.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > ./nist_lm/bigram.arpa.gz
~/kaldi/tools/irstlm/src/compile-lm ./lm_tmp/unigram.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > ./nist_lm/unigram.arpa.gz
cd ../lang
./../utils/prepare_lang.sh > L.fst
./../../../timit/s5/local/timit_format_data.sh > G.fst