def format_arc(src,dst,src_sym,dst_sym,w):
    print("{} {} {} {} {} \n".format(src,dst,src_sym,dst_sym,w))

word_corpus = ['asd','uhsoa']
s = 1
a = []

for word in word_corpus:
    format_arc(src=0, dst=s, src_sym="<epsilon>", dst_sym=word[0], w=0)
    for i in range(len(word) -1):
        format_arc(src=s, dst=s+1, src_sym=word[i], dst_sym=word[i+1], w=0)
        s += 1
    format_arc(src=s, dst=s+1, src_sym=word[len(word) -1 ], dst_sym="<epsilon>", w=0)
    a.append(s+1)
    s += 2

for l in a:
    print("{}\n".format(l))
