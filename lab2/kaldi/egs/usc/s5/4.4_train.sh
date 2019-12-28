source ./path.sh
source ./cmd.sh


#Step 1: Train monophone model

echo "-----------------------------------"
echo "Train Monophones"
echo "-----------------------------------"

#- Location of the acoustic data: `data/train` 
#- Location of the lexicon: `data/lang`
#- Source directory for the model: `exp/lastmodel`
#- Destination directory for the model: `exp/currentmodel`
#Since a model does not yet exist, there is no source directory specifically for the model. 

#./steps/train_mono.sh --boost-silence 1.25  --cmd "run.pl" ./data/train ./data/lang_test ./exp/mono


echo "-----------------------------------"
echo "Align Monophones"
echo "-----------------------------------"

#- Destination directory for the alignment: `exp/currentmodel_ali
#./steps/align_si.sh --boost-silence 1.25 ./data/train ./data/lang_test exp/mono exp/mono_ali || exit 1;


#Step 2: Create HCLG  graph using G.fst 

# The below script creates a fully expanded decoding graph (HCLG) that represents
# all the language-model, pronunciation dictionary (lexicon), context-dependency,
# and HMM structure in our model.  The output is a Finite State Transducer
# that has word-ids on the output, and pdf-ids on the input (these are indexes
# that resolve to Gaussian Mixture Models)

echo "-----------------------------------"
echo "Create HCLG graph"
echo "-----------------------------------"

#for x in unigram bigram; do
#  mv ./data/lang_test/G_train_$x.fst ./data/lang_test/G.fst
#  ./utils/mkgraph.sh --mono ./data/lang_test ./exp/mono exp/mono/graph_$x
#  mv ./data/lang_test/G.fst ./data/lang_test/G_train_$x.fst 
#done  
  
  
#Step 3,4: 

#for x in test dev; do
#    for y in unigram bigram; do
#        #Decode test and validation sentences with Viterbi algorithm
#        echo "-----------------------------------"
#        echo "Decode ${x}_${y}"
#        echo "-----------------------------------"
#        ./steps/decode.sh exp/mono/graph_$y ./data/$x ./exp/mono/decode_${x}_${y}
#        #Print PER
#        echo "-----------------------------------"
#        echo "Calculate PER for ${x}_${y}"
#        echo "-----------------------------------"
#        [ -d exp/mono/decode_${x}_${y} ] && grep WER exp/mono/decode_${x}_${y}/wer_* | ./utils/best_wer.sh
#    done;
#done


#Step 5

#Train delta-based triphones
echo "-----------------------------------"
echo "Train Triphones"
echo "-----------------------------------"
./steps/train_deltas.sh --boost-silence 1.25 2000 10000 ./data/train ./data/lang ./exp/mono_ali ./exp/tri1 || exit 1;



echo "-----------------------------------"
echo "Align Triphones"
echo "-----------------------------------"
#Align delta-based triphones
./steps/align_si.sh ./data/train ./data/lang ./exp/tri1 ./exp/tri1_ali || exit 1;

echo "-----------------------------------"
echo "Create HCLG graph"
echo "-----------------------------------"
for x in unigram bigram; do
  mv ./data/lang_test/G_train_$x.fst ./data/lang_test/G.fst
  ./utils/mkgraph.sh ./data/lang_test ./exp/tri1 exp/tri1/graph_$x
  mv ./data/lang_test/G.fst ./data/lang_test/G_train_$x.fst 
done


for x in test dev; do
    for y in unigram bigram; do
        #Decode test and validation sentences with Viterbi algorithm
        echo "-----------------------------------"
        echo "Decode ${x}_${y}"
        echo "-----------------------------------"
        ./steps/decode.sh exp/tri1/graph_$y ./data/$x ./exp/tri1/decode_${x}_${y}
        #Print PER
        echo "-----------------------------------"
        echo "Calculate PER for ${x}_${y}"
        echo "-----------------------------------"
        [ -d exp/tri1/decode_${x}_${y} ] && grep WER exp/tri1/decode_${x}_${y}/wer_* | ./utils/best_wer.sh
    done;
done




