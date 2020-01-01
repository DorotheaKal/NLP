#!/usr/bin/env python
# coding: utf-8

# In[6]:



#!/usr/bin/python3
f = open('./data/local/dict/silence_phones.txt',"w+")
f.write("sil" + '\n')
f.close()

f = open('./data/local/dict/optional_silence.txt',"w+")
f.write("sil"+'\n')
f.close()

nons = open('./data/local/dict/nonsilence_phones.txt',"w+")
lex_data = open('./slp_lab2_data/lexicon.txt',"r+")
lex = open("./data/local/dict/lexicon.txt","w+")
f = open('./data/local/dict/extra_questions.txt',"w+")
f.close()
phones = set()
for line in lex_data:

    line = line.split()[1:]
    
    for phonem in line:
        phones.add(phonem)

phones = sorted(phones)
for phonem in phones:
    if phonem != 'sil':
        nons.write(phonem + '\n')
    lex.write(phonem + ' ' + phonem + '\n')

for name in ['test','dev','train']:
    set_txt = open('./data/'+name+'/text',"r")   
    lm_txt = open('./data/local/dict/lm_'+name+'.text',"w+")
    for line in set_txt:
        line = line.split("sil")
        line.insert(1,'<s> sil')
        line.pop()
        line.append('sil </s>\n')
        for l in line:
            lm_txt.write(l)
    set_txt.close()
    lm_txt.close()        


# In[ ]:




