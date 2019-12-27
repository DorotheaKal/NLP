source ./path.sh
source ./cmd.sh

#Step 4.3 

for x in train test dev; do
        echo "--------------------------------"
        echo "Start make_mfcc"
        echo "--------------------------------"
        ./steps/make_mfcc.sh  --mfcc-config ./conf/mfcc.conf --cmd  "run.pl" --nj 4 data/$x exp/make_mfcc/$x mfcc_${x}
        echo "mfcc_stats for ${x}"
        ./steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x mfcc_stats
done
