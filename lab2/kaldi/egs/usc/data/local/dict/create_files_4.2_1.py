#!/usr/bin/python3
f = open('silence_phones.txt',"w+")
f.write("sil")
f.close()

f = open('optional_silence.txt',"w+")
f.write("sil")
f.close()

# rip data directory
data_dir = '../../../../../../slp_lab2_data'

nons = open('nonsilence_phones.txt',"w+")
lex_data = open(f'{data_dir}/lexicon.txt',"r+")
lex = open("lexicon.txt","w+")
f = open('extra_questions.txt',"w+")
f.close()
phones = set()
for line in lex_data:

    line = line.split()[1:]
    
    for phonem in line:
        phones.add(phonem)

phones = sorted(phones)
for phonem in phones:
    lex.write(phonem + ' ' + phonem + '\n')
    nons.write(phonem + '\n')


for name in ['test','dev','train']:
    set_txt = open(f'../../{name}/text',"r")   
    lm_txt = open(f'lm_{name}.text',"w+")
    for line in set_txt:
        line = line.split("sil")
        line.insert(1,'<s> sil')
        line.pop()
        line.append('sil </s>\n')
        for l in line:
            lm_txt.write(l)
    set_txt.close()
    lm_txt.close()        
