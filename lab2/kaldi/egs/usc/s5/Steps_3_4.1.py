#!/usr/bin/python3
import subprocess
import re
import os


# Soft links
os.system('ln -s  ../../wsj/s5/steps/ steps')
os.system('ln -s  ../../wsj/s5/utils/ utils')
subprocess.call(['mkdir','local'])
os.system('cd ./local')
os.system(' ln -s ../../../wsj/s5/steps/score_kaldi.sh score.sh')
os.system('cd ../')
#subprocess.call(["mv", "./local/score_kaldi.sh", "./local/score.sh"])
# Dir conf
subprocess.call(['mkdir','conf'])
os.system('cp ./lab2_help_scripts/mfcc.conf ./conf/mfcc.conf')
# Copy path.sh and cmd.sh
os.system('cp ./../../wsj/s5/path.sh ./path.sh')
os.system('cp ./../../wsj/s5/cmd.sh ./cmd.sh')


def make_phonems(sentence,phonems):
    sentence = sentence.strip().lower()
    sentence = re.sub("[^a-z^\'\s\-]",'',sentence)
    words = re.split('\s|-',sentence)
    
    res = '' 
    for word in words:
        res = res + ' ' + phonems[word]
        
    res = 'sil ' + res + ' sil'
    return res 

data_dir = './slp_lab2_data'
text_dir = f'{data_dir}/transcription.txt'
phonems_dir = f'{data_dir}/lexicon.txt'

text = open(text_dir,"r")
utterances = []
for line in text:
    utterances.append(line)
text.close()

phons = open(phonems_dir,"r")
phonems = {}
for line in phons:
    line = line.split()
    phonems[line[0].strip().lower()] = ' '.join(word for word in line[1:])

for name in ['test','train','validation']:
    subprocess.call(['mkdir','-p',f'./data/{name}'])
i = 0 


for name in ['test','train','validation']:
    input_dir = f'{data_dir}/filesets/{name}_utterances.txt'
    inp = open(input_dir,"r")
    
    dest_ids=f'./data/{name}/uttids'
    dest_speaker=f'./data/{name}/utt2spk'
    dest_path=f'./data/{name}/wav.scp'
    text_path=f'./data/{name}/text'

    ids = open(dest_ids,"w+")
    speakers = open(dest_speaker,"w+")
    wavs = open(dest_path,"w+")
    utt = open(text_path,'w+')

    for line in inp:
        line = line.strip('\n')
        id = re.sub('[0-9]*$',str(i),line)
        id = re.sub('[m|f][0-9]','',id)
        
        speaker = re.findall('[m|f][0-9]',line)[0]
        id = speaker + '-' + id 
        id = re.sub('_','-',id)
        ids.write(id+'\n')
        
        
        speakers.write(f'{id} {speaker}\n')
        
        wavs.write(f'{id} {data_dir}/wav/{speaker}/{line}.wav\n')
        
        line_num = re.findall('[0-9]*$',line.strip())[0]
        line_num = int(line_num)-1
        sentence = utterances[line_num]
        sentence = make_phonems(sentence,phonems)
        utt.write(f'{id} {sentence}\n')
        i+=1
    speakers.close()
    inp.close()
    wavs.close()
    ids.close()
    utt.close()
    os.system(f'./utils/utt2spk_to_spk2utt.pl ./data/{name}/utt2spk > ./data/{name}/spk2utt')


subprocess.call(['mkdir','data/local'])
subprocess.call(['mkdir','data/lang'])
for name in ['dict','lm_tmp','nist_lm']:
    subprocess.call(['mkdir',f'data/local/{name}'])
os.system('rm -rf   ./data/dev')
os.system('mv ./data/validation ./data/dev')