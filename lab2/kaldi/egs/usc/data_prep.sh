#!/bin/bash
#data dir, change if needed
data_dir='../../../slp_lab2_data'
mkdir ./data
for name in test train validation; do 
    mkdir ./data/$name
done
i=0
for name in test train validation; do
    input=$data_dir/filesets/${name}_utterances.txt
    dest_ids=./data/$name/uttids
    dest_speaker=./data/$name/utt2spk
    dest_path=./data/$name/wav.scp
    rm -f $dest_ids $dest_speaker $dest_path
    while IFS= read -r line
    do
        id="$(echo $line | sed "s/[0-9]*$//g"  )"
        id="${id}${i}"
        echo  $id >> $dest_ids 
        speaker="$(echo $line | grep -o '[m|f][0-9]' )"
        line="$(echo $line | grep -o '[m|f][0-9]' )"
        echo "$id $speaker" >> $dest_speaker
        echo "$id /$data_dir/wav/$line.wav" >> $dest_path
        i=$((i+1))
        

    done < "$input"

done 
