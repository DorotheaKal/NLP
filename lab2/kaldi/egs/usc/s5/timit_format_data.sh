#!/bin/bash

# Copyright 2013  (Author: Daniel Povey)
# Apache 2.0

# This script takes data prepared in a corpus-dependent way
# in data/local/, and converts it into the "canonical" form,
# in various subdirectories of data/, e.g. data/lang, data/train, etc.

. ./path.sh || exit 1;
. ./cmd.sh 

echo "Preparing train, dev and test data"
srcdir=./data/local/data
lmdir=./data/local/nist_lm
tmpdir=./data/local/lm_tmp
lexicon=./data/local/dict/lexicon.txt
mkdir -p $tmpdir

# Next, for each type of language model, create the corresponding FST
# and the corresponding lang_test_* directory.

echo "Preparing language models for test"

for x in train dev test; do
  test=data/lang_test
  mkdir -p $test
  cp -r data/lang/* $test

  gunzip -c $lmdir/lm_${x}_unigram.arpa.gz | \
    arpa2fst --disambig-symbol=#0 \
             --read-symbol-table=$test/words.txt - data/lang_test/G_${x}_unigram.fst
done

for x in train dev test; do
   test=data/lang_test

   gunzip -c $lmdir/lm_${x}_bigram.arpa.gz | \
     arpa2fst --disambig-symbol=#0 \
              --read-symbol-table=$test/words.txt - data/lang_test/G_${x}_bigram.fst
done

echo "Succeeded in formatting data."



