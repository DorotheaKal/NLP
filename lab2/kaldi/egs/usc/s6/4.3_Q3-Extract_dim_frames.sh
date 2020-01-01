. ./path.sh

feat-to-dim ark:./mfcc_train/raw_mfcc_train.1.ark -
feat-to-len ark:./mfcc_train/raw_mfcc_train.1.ark ark,t:data/train/feats.lengths
head -5 data/train/feats.lengths

