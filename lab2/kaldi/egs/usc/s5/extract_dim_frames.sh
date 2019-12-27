source ./path.sh
$KALDI_ROOT/src/featbin/feat-to-dim ark:./mfcc_train/raw_mfcc_train.1.ark ark,t:feat_dim.txt
$KALDI_ROOT/src/featbin/feat-to-len ark:./mfcc_train/raw_mfcc_train.1.ark ark,t:feat_len.txt
