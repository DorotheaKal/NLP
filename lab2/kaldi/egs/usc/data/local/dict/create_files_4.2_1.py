#!/usr/bin/python3
f = open('silence_phones.txt',"w")
f.write("sil")
f.close()

f = open('optional_silence.txt',"w")
f.write("sil")
f.close()

f = open('../../train/text',"r")
f1 = open('nonsilence_phones.txt',"w")
f2 = open('lexicon.txt',"w")
f3 = open('lm_train.text',"w")
f4 = open('extra_questions.txt',"w")
phones = {}
for line in f:
    line1 = line.split("sil  ")[1].split(' ')
    line1.pop()
    for l in line1:
        if l in phones.keys():
            phones[l] += 1
        else: 
            phones[l] = 1 
    line2 = line.split("sil")
    line2.insert(1,'<s> sil')
    line2.pop()
    line2.append('sil </s>\n')
    for l in line2:
        f3.write(l)

final_list = list(phones.keys())
final_list.sort()
f2.write('sil sil\n')
for final in final_list:
    f1.write(final + '\n')
    f2.write(final+' '+final+'\n')
        
f.close()
f1.close()
f2.close()
f3.close()
f4.close()


f1 = f = open('../../test/text',"r")
f2 = f = open('../../dev/text',"r")
f3 = open('lm_test.text',"w")
f4 = open('lm_dev.text',"w")
        
for line in f1:
    line1 = line.split("sil")
    line1.insert(1,'<s> sil')
    line1.pop()
    line1.append('sil </s>\n')
    for l in line1:
        f3.write(l)
        
for line in f2:
    line1 = line.split("sil")
    line1.insert(1,'<s> sil')
    line1.pop()
    line1.append('sil </s>\n')
    for l in line1:
        f4.write(l)

f1.close()
f2.close()
f3.close()
f4.close()