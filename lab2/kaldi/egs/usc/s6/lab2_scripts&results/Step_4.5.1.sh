source ./path.sh
source ./cmd.sh

#Step 1

# ------------------- Data preparation for DNN -------------------- #
# Compute cmvn stats for every set and save them in specific .ark files
# These will be used by the python dataset class that you were given
echo "# ------------------- Data preparation for DNN -------------------- #"
for set in train dev test; do
  ${KALDI_ROOT}/src/featbin/compute-cmvn-stats --spk2utt=ark:data/${set}/spk2utt scp:data/${set}/feats.scp ark:data/${set}/${set}"_cmvn_speaker.ark"
  ${KALDI_ROOT}/src/featbin/compute-cmvn-stats scp:data/${set}/feats.scp ark:data/${set}/${set}"_cmvn_snt.ark"

done

# --------------------- Alignment of validation and test set ----------------- #

echo "# --------------------- Alignment of validation and test set ----------------- #"
workdir=tri1
nj=4
steps/align_si.sh --nj ${nj} --cmd "$train_cmd" data/dev data/lang exp/${workdir} exp/${workdir}_ali_dev
steps/align_si.sh --nj ${nj} --cmd "$train_cmd" data/test data/lang exp/${workdir} exp/${workdir}_ali_test
