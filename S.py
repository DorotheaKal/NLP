def format_arc(src,dst,src_sym,dst_sym,w):
    print("{} {} {} {} {} \n".format(src,dst,src_sym,dst_sym,w))

acceptor = []
s = 1

letters =  ['a','b','c','d']

for i in range(0, len(letters)):
    format_arc(src=0, dst=s, src_sym="<epsilon>", dst_sym=letters[i], w=1)
    format_arc(src=s, dst=s, src_sym=letters[i], dst_sym=letters[i], w=0)
    format_arc(src=s, dst=0, src_sym=letters[i], dst_sym="<epsilon>", w=1)
    k = 1
    for j in range(0, len(letters)):
        if(j!=i):
            format_arc(src=s, dst=k, src_sym=letters[i], dst_sym=letters[j], w=1)
        k += 1
    s += 1
            
