#!/usr/bin/env python
# coding: utf-8

# <h1 align = "center">Επεξεργασία Φωνής και Φυσικής Γλώσσας</h1> 
# <h2 align = "center">1η Προπαρασκευαστική Εργασία</h2> 
# <h3 align = "center"> Θεoδωρόπουλος Νικήτας -03115185</h3>
# <h3 align = "center"> Καλλιώρα Δωροθέα - 03115176</h3>
# 
# 
# 

# ***Σκοπός***
# 
# Στόχος της εργαστηριακής άσκησης είναι η δημιουργία ενός απλού ορθογράφου με χρήση μηχανών πεπερασμένων καταστάσεων (**fst**) με τη βοήθεια της βιβλιοθήκης openfst (v1.6.1). Για την εκπαίδευση του μοντέλου χρησιμοποιούμε corpus απο δημόσια διαθέσιμα βιβλία απο τα οποία με κατάλληλο tokenization γίνεται εξαγωγή λέξεων και σχηματισμός λεξικού. Για κάθε λέξη προς διόρθωση υπολογίζουμε την απόσταση Levenshtein πάνω στο λεξικό, η λέξη με την ελάχιστη απόσταση είναι η πρόβλεψη του μοντέλο μας.
# 
# Τέλος θα γίνει εισαγωγή σε αναπαραστάσεις **word2vec**. Ενα σύνολο μοντέλων που χρησιμοποιούνται για παραγωγή αναπαραστάσεων λέξεων (embeddigns) σε έναν d-διάσταστο διανυσματικό χώρο $\mathbb{R}^d$, έτσι ώστε λέξεις με κοντινή σημασία να βρίσκονται κοντά και στον διανυσματικό χώρο. Η βασική υπόθεση είναι ότι λέξεις με κοινή κατανομή στο κείμενο θα έχουν και κοινή σημασία. 

# ***Βήμα 1***
# 
# 
# Κατασκευάζουμε το corpus με σύμπτηξη plain text βιβλίων, δημόσια διαθέσιμα στο Project Gutenberg. Θα χρησιμοποιήσουμε corpus των δυο παρακάτω βιβλίων:  
# <br>
# <div class="image123">
#     <div class="imgContainer"  Style = "float:left">
#         <p>The War of the Worlds by H. G. Wells</p>
#         <img src="./img/book36.jpg" height=auto width="250"/>
#     </div>
#     <div class="imgContainer" Style = "float:right">
#         <p>Grimms' Fairy Tales by Jacob Grimm and Wilhelm Grimm</p>
#         <img class="middle-img" src="./img/book2591.jpg"/ height=auto width="250"/>
#     </div>
# </div>
# 

# (β) Κατα την κατασκευή γλωσσικών μοντέλων ειναι κοινή πρακτική η σύμπτηξη πολλών βιβλίων για την δημιουργία ενός ενιαίου corpus προς επεξεργασία. Προφανώς για οποιοδήποτε μοντέλο η αύξηση των δεδομένων εκπαίδευσης οδηγεί σε μεγαλύτερη ικανοτητα γενίκευσης. 
# 
# * Για το μοντέλο μας η επιλογή μιας μόνο πηγής δεδομένων εισάγει μεγάλη προκατάληψη (bias) καθώς περιοριζόμαστε στο λεξιλόγιο ενός μόνο συγγραφέα μιας συγκεκριμένης εποχής ή και του context του βιβλίου (όπως η κοινωνική τάξη των χαρακτήρων αν είναι αφηγηματικό). Για να μπορεί να διορθώσει σωστά μια λέξη το μοντέλο μας θα πρέπει πρώτα να την γνωρίζει, συνεπώς η ποικιλία στο λεξιλιλόγιο είναι ενας καθοριστικός παράγωντας για την επιτυχία του μοντέλου. 
# 
# * Σε αναπαραστάσεις που βασίζονται στα συμφραζόμενα, όπως το word2vec, απαιτείται μεγάλο πλήθος δεδομένων έτσι ώστε να προσδιοριστεί σωστά η σημασία μιας λέξης. Αυτο συμβαίνει γιατί πρέπει να αναλυθεί η χρήση της και να αναγνωριστεί η θέση της σε διαφορετικά γλωσσικά περιβάλλοντα και ισχύει ακόμα και για σταθερό λεξιλόγιο. 
# 

# In[61]:


# Get data for step 1

#!wget -P ./data/ http://www.gutenberg.org/files/36/36-0.txt
#!wget -P ./data/ http://www.gutenberg.org/files/2591/2591-0.txt


# ***Βήμα 2***  
# Για την προεπεξεργασία του αρχέιου εισόδου υλοποιούμε κατάλληλο tokenizer ο οποίος αγνοεί τα σημεία στίξης, τους αριθμούς και οποιουσδήποτε άλλους μη λεκτικούς χαρακτήρες. Διαβάζουμε το αρχείο γραμμή προς γραμμή εφαρμόζοντας την συνάρτηση και προκύπτει μια λίστα απο lowercase λέξεις.

# In[62]:


# Step 2

def identity_preprocess(str):
  return str

def readfile(path,preprocess=identity_preprocess):
  processed_txt=[]
  f = open(path, "r")
  for line in f:
    processed_txt = processed_txt + preprocess(line)
  return processed_txt 

def tokenize(s):
    s = s.strip().lower()
    s = ''.join(c if c.isalpha() else ' ' for c in s)
    # We replace all non-alpha characters with ' '
    s = s.replace('\n',' ')
    s = s.split()
    return s


# (δ)
# Ο tokenizer που υλοποιούμε ειναι πολυ απλός και αναγνωρίζει ως tokens μόνο τις απλές λέξεις. Επίσης δεν έχουμε λάβει υπόψην μας ιδιαίτερα γραμματικά φαινόμενα όπως ή σύντμηση ( "did not $\rightarrow$ didn't"). Στην βιλιοθήκη nltk υπάρχουν αρκετά πιο ακριβείς και εκλεπτυσμένοι tokenizers. Παρασουσιάζουμε ενδεικτικά παρακάτω τα αποτέλσματα για την πρόταση:
# 
# _"At eight o'clock on Thursday morning....Arthur didn't feel very good. He quickly rushed to the Doctor!"_

# In[63]:


#Step 2 (d)
import nltk
#nltk.download('punkt')

sentence = """At eight o'clock on Thursday morning....Arthur didn't feel very good.
              He quickly rushed to the Doctor!"""

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
tokenizers = [tokenize,nltk.word_tokenize,sent_detector.tokenize]
names = ["custom","nltk punkt","sentece detector"]
for tokenizer,name in zip(tokenizers,names):
    print(name,":\n",tokenizer(sentence),'\n')


# Περιληπτικά αναλύουμε τους tokenizers.
# 
# <b>Sentence Tokenizer</b>: Αυτός ο tokenizer χωρίζει το κείμενο σε μία λίστα από προτάσεις χρησιμοποιώντας unsupervised learning για να αναγνωρίσει επιτυχώς τα διαχωριστηκά σημεία στίξης, και να αγνοείσει όπως η απόστροφος σε μία λέξη.  Ο αλγόριθμος αυτός πρέπει να εκπαιδευτεί σε ένα μεγάλο σύνολο δεδομένων πριν μπορέσει να χρησιμοποιηθεί αποτελεσματικά. Το NLTK data package περιλαμβάνει ένα προ-εκπαιδευμένο Punkt tokenizer για την αγγλική γλώσσα.
# 
# <b>Word Τokenizer</b>: Αυτός ο tokenizer εντοπίζει τις λέξεις σε ένα string και τις αποθηκεύει σε μία λίστα. Από τα σημεία στίξης κρατάει μόνο εκείνα που διαχωρίζουν προτάσεις. Όπως και ο Sentence Tokenizer χρησιμοποιεί ένα προ-εκπαιδευμένο Punkt tokenizer για την αγγλική γλώσσα. 
# 
# Η δικία μας απλή εκδοχή του tokenizer είναι αρκετά αποτελεσματική και υστερεί μόνο όταν σημεία στίξης αποτελούν μέρος της λέξης.  
# 
# Η βιβλιοθήκη nltk διαθέτει αρκετά εξεζητημένους tokenizers, ενδεικτικά αναφέρουμε τον tweet tokenizer που διατηρεί τα σημεία στίξης και τα emoji ενω αναγνωρίζει και τα hashtags. Ολα αυτά έχουν μεγάλη πληροφορία για εφαρμογές ανάλυσης tweet.

# ***Βήμα 3*** 
# Με βάση τα tokens που προκύπτουν απο την προηγούμενη ανάλυηση βρίσκουμε τά μοναδικά tokens (λεξιλόγιο) και τους μοναδικούς χαρακτήρες (αλφάβητο) στο corpus.

# In[64]:


# Step 3


text_1 = readfile('./data/36-0.txt',tokenize)
text_2 = readfile('./data/2591-0.txt',tokenize)
text = text_1 + text_2

print(len(text),"words overall")
word_corpus = list(set(text))
print(len(word_corpus),"unique words in corpus")

# Convert list of words to list of chars, get unique chars with set
alphabet = list(set([c for word in word_corpus for c in word]))
print(len(alphabet),"symbols in alphabet:")
print(alphabet)


# Παρατηρούμε οτι το αλφάβητο είναι επαυξημένο με παραλλαγές γραμμάτων και έχει μέγεθος 28.

# In[65]:


# Step 4

def create_syms(alphabet):
    f = open("./chars.syms","w+")
    f.write(f'<epsilon>     0\n')  #for <epsilon> index 0
    for i in range(len(alphabet)):
        f.write(f'{alphabet[i]}     {i+50}\n')  #for other characters index i+50
        
create_syms(alphabet)
    


# ***Βήμα 5*** 
# Για την δημιουργία ενός ορθογράφου απαιτείται κατάλληλη μετρική για την απόσταση δύο λέξεων. 
# 
# θα χρησιμοποιήσουμε την απόσταση Levenshtein (ή edit distance). Η απόσταση μπορέι να υπολογιστεί αναδρομικά με χρήση dynamic programming (DP). Ανάμεσα σε δύο λέξεις μπορούν να γίνουν τρείς τύποι μετατροπών: εισαγωγή ενός χαρακτήρα ($ \epsilon \rightarrow a$), μετατροπή ενός χαρακτήρα σε έναν άλλο ($ a \rightarrow b$) και διαγραφή ενός χαρακτήρα ($a \rightarrow \epsilon$). Θεωρούμε αρχικά ίδιο κόστος για όλες τις μετατροπές.
# 
# (α) Δημιουργούμε κατάλληλο fst με μία κατάσταση, αντιστοιχίζοντας κάθε χαρακτήρα σε κάθε άλλον και στο $\epsilon$, και το $\epsilon$ σε κάθε χαρακτήρα με βάρος 1. Αντιστοιχίζουμε ακόμα τον χαρακτήρα στον εαυτό του με βάρος 0. Αν πάρουμε το shortest path τότε η έξοδος θα είναι ή ίδια η λέξη εισόδου, αφού το ελάχιστο βάρος προκύπτει αν δεν γίνει καμία μετατροπή. 
# 
# (β) Στην υλοποίηση που έγινε για το ερώτημα 5 έχουμε υποθέσει ότι όλα τα edits έχουν ίσο βάρος. Αυτό ουσιαστικά σημαίνει ότι για το μοντέλο μας οποιοδήποτε λάθος σε μία λέξη έχει την ίδια πιθανότητα εμφάνισης. Με βάση την διαίσθηση μας μια τέτοια θεώρηση δεν ανταποκρίνεται στα πραγματικά λάθη που συναντάμε σε κείμενα και προκύπτουν απο τον άνθρωπο. Για παράδειγμα μια λέξη που ξεκινά απο $b$ είναι σχεδόν αδύνατον να γραφεί λανθασμένα με $c$. Ιδανικά θα θέλαμε να γνωρίζουμε την κατανομή του λάθους για κάθε σύμβολο στο αλφάβητο η οποία εδώ έχουμε υποθέσει οτι ειναι ομοιόμορφη. 
# 
# Για να γίνει πειραματικός υπολογισμός θα θέλαμε ενα σύνολο δεδομένων train data της μορφής: $(original~word, wrong~spelling)$. Απο αυτό μπορούμε να υπολογίσουμε κάθε φορά την διόρθωση που απαιτείται. Η πιθανότητα θα προκύψει:
# 
# $$ Pr[ a\rightarrow b] = \frac{|error = b \rightarrow a|}{|train~samples|}, \forall a,b \in \{A+\epsilon\} $$, όπου $A$ το αλφάβητο. 
# 
# Αυτές εινα οι a priori πιθανότητες για κάθε διόρθωση με βάση το training data. Μπορούν τώρα να χρησιμοποιηθούν για να υπολογιστούν τα βάρη στο μοντέλο αναγνώρισης μας με fst. Το βάρος για μια συγκεκριμένη διόρθωση Θα πρέπει να είναι αντιστρόφως ανάλογο της πιθανότητας εμφανίσης του αντίστοιχου λάθους. Η καθιερωμένη συνάρτηση αντιστοίχισης είναι η $ -log(p(X)) $ όπου $p(X)$ η συνάρτηση κατανομής πιθανοτήτων.
# 
# 

# In[66]:


#Step 5


f = open('lev.fst',"w+")
# create a line corresponding to fst edge.
def format_arc(src,dst,src_sym,dst_sym,w,f,acceptor=False):
    if not acceptor:
        f.write("{} {} {} {} {} \n".format(src,dst,src_sym,dst_sym,w))
    else:
        f.write("{} {} {} \n".format(src,dst,src_sym))
letters =  alphabet

for i in range(0, len(letters)):
    format_arc(src=0, dst=0, src_sym="<epsilon>", dst_sym=letters[i], w=1,f=f)
    format_arc(src=0, dst=0, src_sym=letters[i], dst_sym=letters[i], w=0,f=f)
    format_arc(src=0, dst=0, src_sym=letters[i], dst_sym="<epsilon>", w=1,f=f)
    for j in range(0, len(letters)):
        if(j!=i):
            format_arc(src=0, dst=0, src_sym=letters[i], dst_sym=letters[j], w=1,f=f)   
f.write('0\n')
f.close()
    


# In[67]:


get_ipython().system('fstcompile --isymbols=chars.syms --osymbols=chars.syms  lev.fst lev.bin.fst')


# ***Βήμα 6***
# 
# Ένας <b>αποδοχέας (acceptor)</b> είναι ένα fst όπου κάθε μετάβαση έχει ενα label (και προεραιτικά βάρος). Στην βιβλιοθήκη openfst1-6-1 μπορέι να θεωρηθεί ως μετατροπέας(transducer) με ίδιο input και output label. 
# 
# Σε αυτό το ερώτημα κατασκευάζουμε εναν αποδοχέα που αποδέχεται κάθε λέξη του λεξικού. Ο αποδοχέας έχει μία κοινή αρχική κατάσταση και απο εκεί κάθε λέξη επεκτείνεται σε καταστάσεις ανεξάρτητα απο τις άλλες. Δεν έχουμε βάρος στις μεταβάσεις (w = 0).

# In[68]:


#Step 6

def create_acceptor(words,file):
    f = open(file,"w+")
    s = 0
    final_states = []
    for word in words:
        format_arc(0,s+1,word[0],word[0],0,f,acceptor=True)
        s += 1
        for letter in word[1:]:
            format_arc(s,s+1,letter, letter, 0,f,acceptor=True)
            s += 1
        final_states.append(s)

    for state in final_states:
        f.write(f'{state}\n')
    f.close()    
create_acceptor(word_corpus,"acceptor.fst")


# In[69]:


get_ipython().system('fstcompile --isymbols=chars.syms --osymbols=chars.syms --acceptor acceptor.fst acceptor.bin.fst')


# (β) Καλούμε παρακάτω τις συναρτήσεις $fstrmepsilon, fstdeterminze, fstminimize$ που βελτιώνουν το μοντέλο μας. Επεξηγούμε συνοπτικά την λειτουργία τους:
# 
# **fstdetermize**: Δέχεται ως είσοδο έναν μετροπέα (transducer) και το αποτέλεσμα είναι ένα ισοδύναμο fst με την ιδιότητα ότι δεν υπάρχει κατάσταση όπου δύο μεταβάσεις έχουν το ίδιο σύμβολο εισόδου. Το αυτόματο γίνεται δηλαδή ντετερμινιστικό ως προς την είσοδο. Η συνάρτηση έχει το μειονέκτημα οτι χρησιμοποιεί το $\epsilon$ σεμεταβάσεις, θεωρώντας το στοιχείου του αλφαβήτου. Ακόμα αν το αρχικό αυτόματο περιέχει $\epsilon$-μεταβάσεις μπορεί το αποτέλεσμα να μην είναι ντετερμινιστικό. 
# 
# **fstrmepsilon**: Αφαιρεί απο ένα αυτόματο όλες τις $\epsilon$-μεταβάσεις (όταν δηλαδή input = output = $\epsilon$). 
# 
# **fstminimize**: Για εναν acceptor η συνάρτηση παράγει το ελάχιστο ισοδύναμο αυτόματο. Για εναν transducer η ελαχιστότητα δεν μπορεί να επιτευχθεί με την αυστηρή έννοια διότι κάτι τέτοιο θα απαιτούσε output labels με την μορφή συμβολοσειρών που δεν υποστηρίζεται απο την fst. Κάθε τέτοια μετάβαση ειναι ανταυτού μια ακολουθία απο μεταβάσεις με έξοδο χαρακτήρα. 
# 

# In[70]:


#Step 6b

get_ipython().system('fstdeterminize acceptor.bin.fst acceptor.bin.fst')
get_ipython().system('fstrmepsilon acceptor.bin.fst acceptor.bin.fst')
get_ipython().system('fstminimize acceptor.bin.fst acceptor.bin.fst')


# ***Βήμα 7***
# 
# Για να υλοποιήσουμε τον ορθογράφο ελάχιστης απόστασης (min edit distance spell checker) θα συνθέσουμε τον Levenshtein transducer με τον αποδοχέα του λεξικού που υλοποιήσαμε σε προηγούμενο ερώτημα. Το αποτέλεσμα είναι ένας transducer που διορθώνει τις λέξεις μόνο με κριτήριο τις ελάχιστες δυνατές μετατροπές που απαιτούνται, χωρίς να λαμβάνει υπόψη του καμία γλωσσική πληροφορία. 
# 
# α) 
# 
# * Για ίσα βάρη στα edits όπως αναλύσαμε και σε προηγούμενα ερωτήματα, όλες οι μετατροπές μεταξύ γραμμάτων έχουν ίση πιθανοτηα και αρα οι αντίστοιχες ακμές ίσο βάρος στο fst. Δεν υπάρχει συνεπώς κάποιο bias προς μια συγκεκριμένη κατέυθυνση και η επιλογή γίνεται καθαρά με την ελάχιστη edit distance. Αυτο μπορεί να οδηγήσει σε λάθη παρόλο που η λέξη προς διόρθωση μπορεί να είναι γνωστή. 
# 
# 
# *  Για διαφορετικά βάρη των edits το fst είναι προδιαθετιμένο κάθε φορά να ακολουθήσει ένα συγκεκριμένο μονοπάτι μεταβάσεων. Αυτη η προδιάθεση μειώνει την τυχαιότητα στην εκτέλεση του μοντέλου, καθώς πολυ συχνά η λέξη με την ελάχιστη απόσταση θα είναι μοναδική. Η εισαγωγή bias με προσεκτική επιλογή των βαρών μπορεί να οδηγήσει σε πολυ καλά αποτελέσματα. 
# 

# In[71]:


#Step 7
#mini.fst - lev.fst

get_ipython().system('fstarcsort --sort_type=olabel lev.bin.fst lev.bin.fst')
get_ipython().system('fstarcsort --sort_type=ilabel acceptor.bin.fst acceptor.bin.fst')


# In[72]:


#Step 7(a)

get_ipython().system('fstcompose  lev.bin.fst acceptor.bin.fst spell_checker.bin.fst')


# In[73]:


#Step 7b
word = ['cit']
create_acceptor(word,"in.fst")

get_ipython().system('fstcompile  --isymbols=chars.syms --osymbols=chars.syms  --acceptor in.fst in.bin.fst')
get_ipython().system('fstarcsort --sort_type=ilabel spell_checker.bin.fst spell_checker.bin.fst ')
get_ipython().system('fstarcsort --sort_type=olabel in.bin.fst in.bin.fst ')
print(f"Min distance prediction for {word} is: ")
get_ipython().system('fstcompose in.bin.fst spell_checker.bin.fst |fstshortestpath --nshortest=1 | fstrmepsilon |  fsttopsort |fstprint -isymbols=chars.syms  -osymbols=chars.syms| cut -f4 | grep -v "<epsilon>" |head -n -1 | tr -d \'\\n\'')


# In[74]:


get_ipython().system('fstcompose in.bin.fst spell_checker.bin.fst |fstshortestpath --nshortest=10   | fstrmepsilon |  fsttopsort | fstdraw --isymbols=chars.syms --osymbols=chars.syms -portrait  | dot -Tjpg > ./img/min_cit.jpg')


# (β) 
# Παρουσιάζουμε σε μορφή διαγράμματος 10 πιθανές έλάχιστες προβλέψεις για την λέξη _"cit"_. Παρατηρούμε ότι λόγω των ίσων βαρών υπάρχουν πολλές διαφορετικές προβλέψεις ελάχιστης απόστασης. Συνεπώς κατω απο αυτές τις συνθήκες είναι αδύνατον το μοντέλο μας να εκτελεί με συνέπεια καλές προβλέψεις. 
# 
# Οι πιθανές προβλέψεις προκύπτουν: 
# 
# $$\{ cat, cut, bit, pit, city, sit, hit, fit, lit, it\}$$ 
# ![predictions-cit](./img/min_cit.jpg)

# ***Βήμα 8*** 
# 
# Θα αξιολογήσουμε την επίδοση του ορθογράφου μας πάνω σε ένα σύνολο δεδομένων για evaluation. Η εκτίμηση μας είναι η λέξη του λεξικού με την ελάχιστη απόσταση απο την λέξη προς διόρθωση. Όπως αναφέραμε η λέξη αυτή δεν είναι μοναδική, και αρα το αποτέλεσμα εμπεριέχει τυχαιότητα. 

# In[75]:


#!wget https://raw.githubusercontent.com/georgepar/python-lab/master/spell_checker_test_set -P ./data/


# In[76]:


# Step 8:
import random
random.seed()

file = open('./data/spell_checker_test_set',"r")
y_test = []
X_test = []
for line in file:
    [y,X] = line.split(':')
    X = X.split()
    for word in X:
        X_test.append(word)
        y_test.append(y)
# Get 20 random words from test set, along with their labels
idxs = random.sample(range(0, len(y_test)), 20)
X_rand = [X_test[i] for i in idxs]
Y_rand = [y_test[i] for i in idxs]


# In[77]:


def predict(Y,X):
    correct_pred=0
    for y,x in zip(Y,X):
        create_acceptor([x],"input.fst")

        get_ipython().system('fstcompile  --isymbols=chars.syms --osymbols=chars.syms  --acceptor input.fst input.bin.fst')
        get_ipython().system('fstarcsort --sort_type=ilabel spell_checker.bin.fst spell_checker.bin.fst ')
        get_ipython().system('fstarcsort --sort_type=olabel input.bin.fst input.bin.fst ')
        prediction = get_ipython().getoutput('fstcompose input.bin.fst spell_checker.bin.fst |fstshortestpath --nshortest=1         | fstrmepsilon |  fsttopsort |fstprint -isymbols=chars.syms  -osymbols=chars.syms        | cut -f4 | grep -v "<epsilon>" |head -n -1 | tr -d \'\\n\' ')
        print("Input:",x,"\n  --Correct:   ",y,"\n  --Prediction:",prediction[0],"\n")
        if y == prediction[0]:
            correct_pred+=1
    print(f"Accuracy:{correct_pred/len(Y)}%")


# In[78]:


predict(Y_rand,X_rand)


# Διορθώνουμε 20 τυχαίες λέξεις στο test set και μετράμε το accuracy.
# 
# Για διαφορετικές επαναλήψεις παρατηρήσαμε οτι η ακρίβεια προκύπτει απο 0.4% εώς 0.7%. Για λέξεις με μεγάλο μήκος που απέχουν πολύ απο άλλες λέξεις στο λεξιλόγιο το μοντέλο έχει καλή απόδοση και τις αναγνωρίζει με επιτυχία. Για παράδειγμα οι λέξεις: $\{ biscuits, independent, bicycle, southern, scissors, visitors \}$ αναγνωρίζονται σωστά. Λέξεις όμως όπως τα: $\{ poems, cake , awful \}$ αναγνωρίζονται δυσκολότερα γιατί ειναι μικρές και έχουν κοινά προθέματα και επιθέματα με άλλες λέξεις. 
# 
# Δύο είναι οι κύριοι παράγοντες προς βελτίωση που επηρεάζουν την απόδοση του μοντέλου:
# 
# * Το μικρό σύνολο εκπαίδευσης. Εάν το μοντέλο δεν γνωρίζει μια λέξη δεν μπορεί να την αναγνωρίσει και αρα η διόρθωση σε αυτή θα προκύπτει πάντα λάθος. Το corpus απο ένωση 2 βιβλίων δεν ειναι αρκετό για να εξαλείψει σε ικανοποιητικό βαθμό αυτον τον τύπο λάθους.
# 
# * Τα ίσα βάρη στις μετατροπές. Σε πολλές περιπτώσεις ακόμα και αν το μοντέλο γνωρίζει μια λέξη δεν καταφέρνει να διορθώσει σε αυτήν γιατί υπάρχουν ακόμα πολλές λέξεις με την ίδια ελάχιστη απόσταση. Με την "δίκαιη" αυτή αντιμετώπιση εισάγεται τυχαιότητα στην απόδοση του μοντέλου καθώς το αν θα προκύψει η σωστή λέξη απο αυτές με την ελάχιστη απόσταση ειναι κατα βάση τυχαίο. 
# 

# ***Βήμα 9*** 
# 
# Στο ερώτημα αυτο θα ασχοληθούμε με αναπαραστάσεις word2vec. Έχουμε ήδη μιλήσει για διανυσματικές αναπαραστάσεις λέξεων στην εισαγωγή. Στόχος είναι η αναπαράσταση λέξεων στον $\mathbb{R}^d$ έτσι ώστε να βρίσκονται σημασιολογικά κοντά. Το training γίνεται με βάση την θέση τους στο κείμενο με κυλιόμενο παράθυρο. Οι 2 βασικές προσεγγίσεις είναι Continuous Bag of Words (η θέση στο παράθυρο δεν εχει σημασία) και continuous skip gram ( η λέξη χρησιμοποείται για πρόβλεψη των γειτονικών). 
# 
# Διαβάζουμε αρχικά το corpus σε μία λίστα απο προτάσεις με το tokenization που είχαμε υλοποιείσει σε προηγούμενο ερώτημα. Στην συνέχεια εκπαιδεύουμε 100 διάστατα word2vec embeddings με βάση τις προτάσεις που προκύπτουν. Χρησιμοποιούμε $window=5$ και $epochs=100$.
# 
# Για 10 τυχαίες λέξεις θα δείξουμε τις σημασιολογικά κοντινότερες τους.

# In[79]:


#Step 9(a)

def tokenized_list(path,preprocess=identity_preprocess):
  list_of_sentences=[]
  f = open(path, "r")
  for line in f:
    l = preprocess(line)
    if l:
        list_of_sentences.append(l)
  return list_of_sentences


list1 = tokenized_list('./data/36-0.txt',tokenize)
list2 = tokenized_list('./data/2591-0.txt',tokenize)
final_list = list1 + list2 


# In[80]:


#!pip install -U gensim
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec


# In[81]:


#Step 9(b,c)

# Initialize word2vec. Context is taken as the 2 previous and 2 next words
model = Word2Vec(final_list, window=5, size=100, workers=4)
model.train(final_list, total_examples=len(final_list), epochs=1000)
# get ordered vocabulary list
voc = model.wv.index2word
# get vector size
dim = model.vector_size
#pick 10 random words from the dictionary
idxs = random.sample(range(0, len(voc)), 10)
rand_words = [voc[i] for i in idxs]
for word in rand_words:
    print(f'Most similar words to "{word}":')
    for word,sim in model.wv.most_similar(word):
        print(f'     "{word}" -- sim: {sim}')


# 
# (γ) Τα αποτελέσματα του μοντέλου δεν ειναι ικανοποιητικα. Οι λέξεις δεν έχουν επι το πλείστον σημασιολογική συσχέτιση, εκτός ενδεχομένως απο κάποιο κοινό θέμα.´Όπως οι μέρες της εβδομάδας, οι αριθμοί, κάποια ανθρώπινη δράση. 
# 
# Θα προσπαθήσουμε τώρα να βελτιώσουμε το αποτέλεσμα του μοντέλου αλλάζωντας τις εξής παραμέτρους:
# * Αυξάνουμε το μέγεθος του παραθύρου context κρατώντας τον αριθμό εποχών σταθερό
# 
# * Αυξάνουμε τον αριθμό των εποχών κρατώντας το μέγεθος του παραθύρου σταθερό
# 
# * Αυξάνουμε και τον αριθμό εποχών και το μέγεθος του παραθύρου
# 
# Παρακάτω παραθέτουμε τα αποτελέσματα του similarity 20 τυχαίων λέξεων για κάθε μία από τις τρεις περιπτώσεις. Για μέγεθος παραθύρου κοντά στο 1 έχουμε πληροφορία σύνταξης. Για μεγαλύτερο παράθυρο αντιστοιχίζουμε λέξεις με βάση την σημασιολογία. 
# 
# Για μεγαλύτερο μέγεθος παραθύρου θα περιμέναμε λοιπόν καλύτερα αποτελέσματα, εφόσων αναζητούμε σημασιολογικά κοντινές λέξεις. Κάτι τέτοιο δεν συμβαίνει στην πράξη. Εικάζουμε οτι αυτο οφείλεται στο μικρό σύνολο δεδομένων εκπαίδευσης που δεν επιτρέπουν στο μοντέλο να έχει μεγάλη εκφραστικότητα. Συγκεκριμένα για ακριβείς αναπαραστάσεις word2vec θέλουμε εκαττομύρια λέξεις και όχι της ταξής των 3000 που προκύπτουν απο το corpus μας. Παρατηρήσαμε οτι το word2vec όπως ειναι υλοποιημένο στην βιβλιοθήκη αγνοεί σπάνιες λέξεις, για αυτο και υπήρξε μείωση στο αρχικό vocabulary που εξάγαμε. 
# 
# Για αύξηση των αριθμών των epochs επίσης δεν έχουμε καλύτερο αποτέλεσμα. Για να επηρεάσει ουσιαστικά ο αριθμός εποχών θα πρέπει να έχουμε ενα αρκετά μεγάλο σύνολο δεδομένων ώστε ο αλγόριθμος να μην κανει πρόωρα converge. Για εμάς και μικρός αριθμός εποχών ειναι αρκετος. 
# 
# Συμπερασματικά  ο πιο καθοριστικός παράγοντας για σωστή εξαγωγή αναπαραστάσεων ειναι ο αριθμός των δεδομένων εκπαίδευσης.
# 

# In[60]:


#Step 9(c (i))

def similarity(w,s,e):
    model = Word2Vec(final_list, window=w, size=s, workers=4)
    model.train(final_list, total_examples=len(final_list), epochs=e)
    voc = model.wv.index2word
    dim = model.vector_size
    rand_words = [voc[i] for i in idxs]
    for word in rand_words:
        print(f'Most similar words to "{word}":')
        for word,sim in model.wv.most_similar(word):
            print(f'     "{word}" -- sim: {sim}')
            


# In[33]:


similarity(10,100,2000)


# In[34]:


similarity(15,100,2000)


# In[31]:


#Step 9(c (ii))

similarity(5,100,2000)


# In[32]:


similarity(5,100,3000)


# In[35]:


#Step 9(c (iii))

similarity(10,100,2000)

