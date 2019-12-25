./../utils/prepare_lang.sh 
Usage: utils/prepare_lang.sh <dict-src-dir> <oov-dict-entry> <tmp-dir> <lang-dir>
e.g.: utils/prepare_lang.sh data/local/dict <SPOKEN_NOISE> data/local/lang data/lang
<dict-src-dir> should contain the following files:
 extra_questions.txt  lexicon.txt nonsilence_phones.txt  optional_silence.txt  silence_phones.txt
See http://kaldi-asr.org/doc/data_prep.html#data_prep_lang_creating for more info.
options: 
<dict-src-dir> may also, for the grammar-decoding case (see http://kaldi-asr.org/doc/grammar.html)
contain a file nonterminals.txt containing symbols like #nonterm:contact_list, one per line.
     --num-sil-states <number of states>             # default: 5, #states in silence models.
     --num-nonsil-states <number of states>          # default: 3, #states in non-silence models.
     --position-dependent-phones (true|false)        # default: true; if true, use _B, _E, _S & _I
                                                     # markers on phones to indicate word-internal positions. 
     --share-silence-phones (true|false)             # default: false; if true, share pdfs of 
                                                     # all silence phones. 
     --sil-prob <probability of silence>             # default: 0.5 [must have 0 <= silprob < 1]
     --phone-symbol-table <filename>                 # default: ""; if not empty, use the provided 
                                                     # phones.txt as phone symbol table. This is useful 
                                                     # if you use a new dictionary for the existing setup.
     --unk-fst <text-fst>                            # default: none.  e.g. exp/make_unk_lm/unk_fst.txt.
                                                     # This is for if you want to model the unknown word
                                                     # via a phone-level LM rather than a special phone
                                                     # (this should be more useful for test-time than train-time).
     --extra-word-disambig-syms <filename>           # default: ""; if not empty, add disambiguation symbols
                                                     # from this file (one per line) to phones/disambig.txt,
                                                     # phones/wdisambig.txt and words.txt
