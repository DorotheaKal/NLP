. ./path.sh
. ./cmd.sh

#Step 4.3 
./utils/fix_data_dir.sh ./data/train
./utils/fix_data_dir.sh ./data/test
./utils/fix_data_dir.sh ./data/dev

rm -rf ./data/train/.backup
rm -rf ./data/test/.backup
rm -rf ./data/dev/.backup

for x in train test dev; do
        
        echo "--------------------------------"
        echo "Start make_mfcc"
        echo "--------------------------------"
        ./steps/make_mfcc.sh  --mfcc-config ./conf/mfcc.conf --cmd  "run.pl" --nj 4 data/$x exp/make_mfcc/$x mfcc_${x}
        echo "mfcc_stats for ${x}"
        ./steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x mfcc_stats
done
