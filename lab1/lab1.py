#!/usr/bin/env python
# coding: utf-8

# <h1 align = "center">Επεξεργασία Φωνής και Φυσικής Γλώσσας</h1> 
# <h2 align = "center">1η Εργασία</h2> 
# <h3 align = "center"> Θεoδωρόπουλος Νικήτας -03115185</h3>
# <h3 align = "center"> Καλλιώρα Δωροθέα - 03115176</h3>
# 
# 
# 

# ### Σκοπός 
# 
# Στόχος της εργαστηριακής άσκησης είναι η δημιουργία ενός απλού ορθογράφου με χρήση μηχανών πεπερασμένων καταστάσεων (**fst**) με τη βοήθεια της βιβλιοθήκης openfst (v1.6.1). Για την εκπαίδευση του μοντέλου χρησιμοποιούμε corpus απο δημόσια διαθέσιμα βιβλία απο τα οποία με κατάλληλο tokenization γίνεται εξαγωγή λέξεων και σχηματισμός λεξικού. Για κάθε λέξη προς διόρθωση υπολογίζουμε την απόσταση Levenshtein πάνω στο λεξικό, η λέξη με την ελάχιστη απόσταση είναι η πρόβλεψη του μοντέλο μας.
# 
# Τέλος θα γίνει εισαγωγή σε αναπαραστάσεις **word2vec**. Ενα σύνολο μοντέλων που χρησιμοποιούνται για παραγωγή αναπαραστάσεων λέξεων (embeddigns) σε έναν d-διάσταστο διανυσματικό χώρο $\mathbb{R}^d$, έτσι ώστε λέξεις με κοντινή σημασία να βρίσκονται κοντά και στον διανυσματικό χώρο. Η βασική υπόθεση είναι ότι λέξεις με κοινή κατανομή στο κείμενο θα έχουν και κοινή σημασία. 

# ### Βήμα 1
# 
# Κατασκευάζουμε το corpus με σύμπτηξη plain text βιβλίων, δημόσια διαθέσιμα στο Project Gutenberg. Θα χρησιμοποιήσουμε corpus των τριών παρακάτω βιβλίων:  
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
#     <div class="imgContainer" Style = "float:middle">
#         <p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
#             Pride and Prejudice by Jane Austen </p>
#         <img src="./img/book1342.jpg"/ height=auto width="250" align="center"/>
#     </div>
# </div>
# 

# (β) Κατα την κατασκευή γλωσσικών μοντέλων ειναι κοινή πρακτική η σύμπτηξη πολλών βιβλίων για την δημιουργία ενός ενιαίου corpus προς επεξεργασία. Προφανώς για οποιοδήποτε μοντέλο η αύξηση των δεδομένων εκπαίδευσης οδηγεί σε μεγαλύτερη ικανοτητα γενίκευσης. 
# 
# * Για το μοντέλο μας η επιλογή μιας μόνο πηγής δεδομένων εισάγει μεγάλη προκατάληψη (bias) καθώς περιοριζόμαστε στο λεξιλόγιο ενός μόνο συγγραφέα μιας συγκεκριμένης εποχής ή και του context του βιβλίου (όπως η κοινωνική τάξη των χαρακτήρων αν είναι αφηγηματικό). Για να μπορεί να διορθώσει σωστά μια λέξη το μοντέλο μας θα πρέπει πρώτα να την γνωρίζει, συνεπώς η ποικιλία στο λεξιλιλόγιο είναι ενας καθοριστικός παράγωντας για την επιτυχία του μοντέλου. 
# 
# * Σε αναπαραστάσεις που βασίζονται στα συμφραζόμενα, όπως το word2vec, απαιτείται μεγάλο πλήθος δεδομένων έτσι ώστε να προσδιοριστεί σωστά η σημασία μιας λέξης. Αυτο συμβαίνει γιατί πρέπει να αναλυθεί η χρήση της και να αναγνωριστεί η θέση της σε διαφορετικά γλωσσικά περιβάλλοντα και ισχύει ακόμα και για σταθερό λεξιλόγιο. 
# 

# In[2]:


# Get data for step 1

#!wget -P ./data/ http://www.gutenberg.org/files/36/36-0.txt
#!wget -P ./data/ http://www.gutenberg.org/files/2591/2591-0.txt
#!wget -P ./data/ http://www.gutenberg.org/files/1342/1342-0.txt


# ### Βήμα 2 
# 
# Για την προεπεξεργασία του αρχέιου εισόδου υλοποιούμε κατάλληλο tokenizer ο οποίος αγνοεί τα σημεία στίξης, τους αριθμούς και οποιουσδήποτε άλλους μη λεκτικούς χαρακτήρες. Διαβάζουμε το αρχείο γραμμή προς γραμμή εφαρμόζοντας την συνάρτηση και προκύπτει μια λίστα απο lowercase λέξεις.

# In[4]:


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

# In[5]:


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
# Η βιβλιοθήκη nltk διαθέτει αρκετά εξεζητημένους tokenizers, ενδεικτικά αναφέρουμε τον tweet tokenizer που διατηρεί τα σημεία στίξης και τα emoji ενω αναγνωρίζει και τα hashtags. Ολα αυτά διαθέτουν μεγάλη πληροφορία, χρήσιμη για εφαρμογές ανάλυσης tweet.

# ### Βήμα 3
# Με βάση τα tokens που προκύπτουν απο την προηγούμενη ανάλυηση βρίσκουμε τά μοναδικά tokens (λεξιλόγιο) και τους μοναδικούς χαρακτήρες (αλφάβητο) στο corpus.

# In[6]:


# Step 3


text_1 = readfile('./data/36-0.txt',tokenize)
text_2 = readfile('./data/2591-0.txt',tokenize)
text_3 = readfile('./data/1342-0.txt',tokenize)

text = text_1 + text_2 + text_3

print(len(text),"words overall")
word_corpus = list(set(text))
print(len(word_corpus),"unique words in corpus")

# Convert list of words to list of chars, get unique chars with set
alphabet = list(set([c for word in word_corpus for c in word]))
print(len(alphabet),"symbols in alphabet:")
print(alphabet)


# Παρατηρούμε οτι το αλφάβητο είναι επαυξημένο με παραλλαγές γραμμάτων και έχει μέγεθος 31.

# ### Βήμα 4
# 
# Δημιουργούμε τον πίνακα συμβόλων με βάση το αλφάβητο που υπολογίσαμε στο προηγούμενο βήμα. Το $\epsilon$ αντιστοιχίζεται στο $0$, για τις υπολοιπες αντιστοιχίες ξεκινάμε απο εναν αυθαίρετο αριθμό. 

# In[6]:


# Step 4

def create_syms(alphabet):
    f = open("./chars.syms","w+")
    f.write(f'<epsilon>     0\n')  #for <epsilon> index 0
    for i in range(len(alphabet)):
        f.write(f'{alphabet[i]}     {i+50}\n')  #for other characters index i+50
        
create_syms(alphabet)
    


# ### Βήμα 5 
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

# In[7]:


#Step 5

# create a line corresponding to fst edge.
def format_arc(src,dst,src_sym,dst_sym,w,f):
    f.write("{} {} {} {} {} \n".format(src,dst,src_sym,dst_sym,w))


def create_lev(file,alphabet,w1):
    f = open(file,"w+")
    for i in range(0, len(alphabet)):
        format_arc(src=0, dst=0, src_sym="<epsilon>", dst_sym=alphabet[i], w = w1,f=f)
        format_arc(src=0, dst=0, src_sym=alphabet[i], dst_sym=alphabet[i], w=0,f=f)
        format_arc(src=0, dst=0, src_sym=alphabet[i], dst_sym="<epsilon>", w = w1,f=f)
        for j in range(0, len(alphabet)):
            if(j!=i):
                format_arc(src=0, dst=0, src_sym=alphabet[i], dst_sym=alphabet[j], w = w1,f=f)   
    f.write('0\n')
    f.close()

create_lev('lev.fst',alphabet,1)    


# In[8]:


get_ipython().system('fstcompile --isymbols=chars.syms --osymbols=chars.syms  lev.fst lev.bin.fst')


# ### Βήμα 6 
# 
# Ένας <b>αποδοχέας (acceptor)</b> είναι ένα fst όπου κάθε μετάβαση έχει ενα label (και προεραιτικά βάρος). Στην βιβλιοθήκη openfst1-6-1 μπορέι να θεωρηθεί ως μετατροπέας(transducer) με ίδιο input και output label. 
# 
# Σε αυτό το ερώτημα κατασκευάζουμε εναν αποδοχέα που αποδέχεται κάθε λέξη του λεξικού. Ο αποδοχέας έχει μία κοινή αρχική κατάσταση και απο εκεί κάθε λέξη επεκτείνεται σε καταστάσεις ανεξάρτητα απο τις άλλες. Δεν έχουμε βάρος στις μεταβάσεις (w = 0).

# In[9]:


#Step 6

def create_acceptor(word_corpus,file,weights = {},model = 'Default'):
    
    ''' Create a suitable acceptor
        word_corpus = dictionary of unique words
        file = destination file
        weights = dictionary with weights per word/letter/bigram respectively.
                  In the case of bigrams it is a tuple.
        model = language model type, one of: {Default,Word,Unigram,Bigram}
    '''
    
    f = open(file,"w+")
    s = 0
    final_states = []
    for word in word_corpus:
        
        if model == 'Default':
            w1 = 0
            w2 = 0
        elif model == 'Word':
            w1 = weights[word]
            w2 = 0
        elif model == 'Unigram':
            w1 = 0
        elif model == 'Bigram':
            w1 = 0
            prev_letter = ' '
            
        format_arc(0,s+1,"<epsilon>","<epsilon>",w1,f)
        s += 1
        
        for letter in word[0:]:
            
            if model == 'Unigram':
                w2 = weights[letter]
            
            if model == 'Bigram':
                w2 = weights[(prev_letter,letter)]
            format_arc(s,s+1,letter, letter,w2,f)
            s += 1
            prev_letter = letter 
        final_states.append(s)

    for state in final_states:
        f.write(f'{state}\n')
    f.close()

create_acceptor(word_corpus,"acceptor.fst")


# In[10]:


get_ipython().system('fstcompile --isymbols=chars.syms --osymbols=chars.syms  acceptor.fst acceptor.bin.fst')


# (β) Καλούμε παρακάτω τις συναρτήσεις $fstrmepsilon, fstdeterminze, fstminimize$ που βελτιώνουν το μοντέλο μας. Επεξηγούμε συνοπτικά την λειτουργία τους:
# 
# **fstdetermize**: Δέχεται ως είσοδο έναν μετροπέα (transducer) και το αποτέλεσμα είναι ένα ισοδύναμο fst με την ιδιότητα ότι δεν υπάρχει κατάσταση όπου δύο μεταβάσεις έχουν το ίδιο σύμβολο εισόδου. Το αυτόματο γίνεται δηλαδή ντετερμινιστικό ως προς την είσοδο. Η συνάρτηση έχει το μειονέκτημα οτι χρησιμοποιεί το $\epsilon$ σεμεταβάσεις, θεωρώντας το στοιχείου του αλφαβήτου. Ακόμα αν το αρχικό αυτόματο περιέχει $\epsilon$-μεταβάσεις μπορεί το αποτέλεσμα να μην είναι ντετερμινιστικό. 
# 
# **fstrmepsilon**: Αφαιρεί απο ένα αυτόματο όλες τις $\epsilon$-μεταβάσεις (όταν δηλαδή input = output = $\epsilon$). 
# 
# **fstminimize**: Για εναν acceptor η συνάρτηση παράγει το ελάχιστο ισοδύναμο αυτόματο. Για εναν transducer η ελαχιστότητα δεν μπορεί να επιτευχθεί με την αυστηρή έννοια διότι κάτι τέτοιο θα απαιτούσε output labels με την μορφή συμβολοσειρών που δεν υποστηρίζεται απο την fst. Κάθε τέτοια μετάβαση ειναι ανταυτού μια ακολουθία απο μεταβάσεις με έξοδο χαρακτήρα. 
# 

# In[11]:


#Step 6b

get_ipython().system('fstdeterminize acceptor.bin.fst acceptor.bin.fst')
get_ipython().system('fstrmepsilon acceptor.bin.fst acceptor.bin.fst')
get_ipython().system('fstminimize acceptor.bin.fst acceptor.bin.fst')


# ### Βήμα 7
# 
# Για να υλοποιήσουμε τον ορθογράφο ελάχιστης απόστασης (min edit distance spell checker) θα συνθέσουμε τον Levenshtein transducer με τον αποδοχέα του λεξικού που υλοποιήσαμε σε προηγούμενο ερώτημα. Το αποτέλεσμα είναι ένας transducer που διορθώνει τις λέξεις μόνο με κριτήριο τις ελάχιστες δυνατές μετατροπές που απαιτούνται, χωρίς να λαμβάνει υπόψη του καμία γλωσσική πληροφορία. 
# 
# α) 
# 
# * Για ίσα βάρη στα edits όπως αναλύσαμε και σε προηγούμενα ερωτήματα, όλες οι μετατροπές μεταξύ γραμμάτων έχουν ίση πιθανοτηα και αρα οι αντίστοιχες ακμές ίσο βάρος στο fst. Δεν υπάρχει συνεπώς κάποιο bias προς μια συγκεκριμένη κατέυθυνση και η επιλογή γίνεται καθαρά με την ελάχιστη edit distance. Αυτο μπορεί να οδηγήσει σε λάθη παρόλο που η λέξη προς διόρθωση μπορεί να είναι γνωστή. 
# 
# 
# *  Για διαφορετικά βάρη των edits το fst είναι προδιαθετιμένο κάθε φορά να ακολουθήσει ένα συγκεκριμένο μονοπάτι μεταβάσεων. Αυτη η προδιάθεση μειώνει την τυχαιότητα στην εκτέλεση του μοντέλου, καθώς πολυ συχνά η λέξη με την ελάχιστη απόσταση θα είναι μοναδική. Η εισαγωγή bias με προσεκτική επιλογή των βαρών μπορεί να οδηγήσει σε καλύτερα αποτελέσματα. 
# 

# In[12]:


#Step 7

#Step 7(a)
get_ipython().system('fstarcsort --sort_type=olabel lev.bin.fst lev.bin.fst')
get_ipython().system('fstarcsort --sort_type=ilabel acceptor.bin.fst acceptor.bin.fst')
get_ipython().system('fstcompose  lev.bin.fst acceptor.bin.fst spell_checker.bin.fst')


# In[13]:


#Step 7b
word = ['cit']
create_acceptor(word,"in.fst")

get_ipython().system('fstcompile  --isymbols=chars.syms --osymbols=chars.syms   in.fst in.bin.fst')
get_ipython().system('fstarcsort --sort_type=ilabel spell_checker.bin.fst spell_checker.bin.fst ')
get_ipython().system('fstarcsort --sort_type=olabel in.bin.fst in.bin.fst ')
print(f"Min distance prediction for {word} is: ")
get_ipython().system('fstcompose in.bin.fst spell_checker.bin.fst |fstshortestpath --nshortest=1 | fstrmepsilon |  fsttopsort |fstprint -isymbols=chars.syms  -osymbols=chars.syms| cut -f4 | grep -v "<epsilon>" |head -n -1 | tr -d \'\\n\'')


# In[14]:


get_ipython().system('fstcompose in.bin.fst spell_checker.bin.fst |fstshortestpath --nshortest=10   | fstrmepsilon |  fsttopsort | fstdraw --isymbols=chars.syms --osymbols=chars.syms -portrait  | dot -Tjpg > ./img/min_cit.jpg')


# (β) 
# Παρουσιάζουμε σε μορφή διαγράμματος 10 πιθανές έλάχιστες προβλέψεις για την λέξη _"cit"_. Παρατηρούμε ότι λόγω των ίσων βαρών υπάρχουν πολλές διαφορετικές προβλέψεις ελάχιστης απόστασης. Συνεπώς κατω απο αυτές τις συνθήκες είναι αδύνατον το μοντέλο μας να εκτελεί με συνέπεια καλές προβλέψεις. 
# 
# Οι πιθανές προβλέψεις προκύπτουν: 
# |
# $$\{ ηit, cut, cat, city, bit, lit, sit, pit, wit\}$$ 
# ![predictions-cit](./img/min_cit.jpg)

# ### Βήμα 8 
# 
# Θα αξιολογήσουμε την επίδοση του ορθογράφου μας πάνω σε ένα σύνολο δεδομένων για evaluation. Η εκτίμηση μας είναι η λέξη του λεξικού με την ελάχιστη απόσταση απο την λέξη προς διόρθωση. Όπως αναφέραμε η λέξη αυτή δεν είναι μοναδική, και αρα το αποτέλεσμα εμπεριέχει τυχαιότητα. 

# In[15]:


#!wget https://raw.githubusercontent.com/georgepar/python-lab/master/spell_checker_test_set -P ./data/


# In[10]:


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


# In[17]:


import os
def predict(Y,X,spell_checker,Show = True):
    correct_pred=0
    for y,x in zip(Y,X):
        create_acceptor([x],"input.fst")
        get_ipython().system('fstcompile  --isymbols=chars.syms --osymbols=chars.syms input.fst input.bin.fst')
        get_ipython().system('fstarcsort --sort_type=olabel input.bin.fst input.bin.fst ')
        command = f'''fstcompose input.bin.fst {spell_checker} |fstshortestpath --nshortest=1         | fstrmepsilon |  fsttopsort |fstprint -isymbols=chars.syms  -osymbols=chars.syms        | cut -f4 | grep -v "<epsilon>" |head -n -1 | tr -d '\n' 
        '''
        prediction = os.popen(command).read()
        if Show:
            print("Input:",x,"\n  --Correct:   ",y,"\n  --Prediction:",prediction,"\n")
        if y == prediction:
            correct_pred+=1
    print(f"{spell_checker}-accuracy:{correct_pred/len(Y)}%")


# In[18]:


predict(Y_rand,X_rand,"spell_checker.bin.fst")


# Διορθώνουμε 20 τυχαίες λέξεις στο test set και μετράμε το accuracy.
# 
# Για διαφορετικές επαναλήψεις παρατηρήσαμε οτι η ακρίβεια προκύπτει απο 0.4% εώς 0.7%. Για λέξεις με μεγάλο μήκος που απέχουν πολύ απο άλλες λέξεις στο λεξιλόγιο το μοντέλο έχει καλή απόδοση και τις αναγνωρίζει με επιτυχία. Για παράδειγμα οι λέξεις: $\{ biscuits, independent, bicycle, southern, scissors, visitors \}$ αναγνωρίζονται σωστά. Λέξεις όμως όπως τα: $\{ poems, cake , awful \}$ αναγνωρίζονται δυσκολότερα (λανθασμένα) γιατί ειναι μικρές και έχουν κοινά προθέματα και επιθέματα με άλλες λέξεις. 
# 
# Δύο είναι οι κύριοι παράγοντες προς βελτίωση που επηρεάζουν την απόδοση του μοντέλου:
# 
# * Το μικρό σύνολο εκπαίδευσης. Εάν το μοντέλο δεν γνωρίζει μια λέξη δεν μπορεί να την αναγνωρίσει και αρα η διόρθωση σε αυτή θα προκύπτει πάντα λάθος. Το corpus απο ένωση 3 βιβλίων δεν ειναι αρκετό για να εξαλείψει σε ικανοποιητικό βαθμό αυτον τον τύπο λάθους. Αν και με την προσθήκη 3ου βιβλίου είδαμε αισθητή βελτίωση απο τα 2 βιβλία. 
# 
# * Τα ίσα βάρη στις μετατροπές. Σε πολλές περιπτώσεις ακόμα και αν το μοντέλο γνωρίζει μια λέξη δεν καταφέρνει να διορθώσει σε αυτήν γιατί υπάρχουν ακόμα πολλές λέξεις με την ίδια ελάχιστη απόσταση. Με την "δίκαιη" αυτή αντιμετώπιση εισάγεται τυχαιότητα στην απόδοση του μοντέλου καθώς το αν θα προκύψει η σωστή λέξη απο αυτές με την ελάχιστη απόσταση ειναι κατα βάση τυχαίο. 
# 

# ### Βήμα 9
# 
# Στο ερώτημα αυτο θα ασχοληθούμε με αναπαραστάσεις word2vec. Έχουμε ήδη μιλήσει για διανυσματικές αναπαραστάσεις λέξεων στην εισαγωγή. Στόχος είναι η αναπαράσταση λέξεων στον $\mathbb{R}^d$ έτσι ώστε να βρίσκονται σημασιολογικά κοντά. Το training γίνεται με βάση την θέση τους στο κείμενο με κυλιόμενο παράθυρο. Ο αλγόριθμος είναι unsupervised και ανάλογα με την προσέγγιση έχει διαφορετικά inputs και outputs. Οι 2 βασικές προσεγγίσεις είναι Continuous Bag of Words (input στο νευρωνικό δίκτυο το context της λέξης, προβλέπεται η λέξη) και continuous skip gram (η λέξη χρησιμοποείται για πρόβλεψη των γειτονικών λέξεων). 
# 
# Διαβάζουμε αρχικά το corpus σε μία λίστα απο προτάσεις με το tokenization που είχαμε υλοποιήσει σε προηγούμενο ερώτημα. Στην συνέχεια εκπαιδεύουμε 100 διάστατα word2vec embeddings με βάση τις προτάσεις που προκύπτουν. Χρησιμοποιούμε $window=5$ και $epochs=100$.
# 
# Για 10 τυχαίες λέξεις θα δείξουμε τις σημασιολογικά κοντινότερες τους.

# In[7]:


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
list3 = tokenized_list('./data/1342-0.txt',tokenize)
final_list = list1 + list2 + list3


# In[8]:


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
# (γ) Τα αποτελέσματα του μοντέλου δεν ειναι ικανοποιητικα. Οι λέξεις δεν έχουν επι το πλείστον σημασιολογική συσχέτιση, εκτός ενδεχομένως απο κάποιο κοινό θέμα.´Όπως οι μέρες της εβδομάδας, οι αριθμοί, ανθρώπινα ονόματα ή κάποια ανθρώπινη δράση. 
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
# Για μεγαλύτερο μέγεθος παραθύρου θα περιμέναμε λοιπόν καλύτερα αποτελέσματα, εφόσων αναζητούμε σημασιολογικά κοντινές λέξεις. Κάτι τέτοιο δεν συμβαίνει στην πράξη. Εικάζουμε οτι αυτο οφείλεται στο μικρό σύνολο δεδομένων εκπαίδευσης που δεν επιτρέπουν στο μοντέλο να έχει μεγάλη εκφραστικότητα. Συγκεκριμένα για ακριβείς αναπαραστάσεις word2vec θέλουμε εκαττομύρια λέξεις και όχι της ταξής των 5000 που προκύπτουν απο το corpus μας. Παρατηρήσαμε οτι το word2vec όπως ειναι υλοποιημένο στην βιβλιοθήκη αγνοεί σπάνιες λέξεις, για αυτο και υπήρξε μείωση στο αρχικό vocabulary που εξάγαμε. Συγκεκριμένα η παράμετρος _min_count (int, optional)_ καθορίζει πόσο μικρή συχνότητα πρέπει να έχει μια λέξη για να θεωρηθεί οτι δεν μπορεί να δώσει με ακρίβεια πληροφορία, και να αγνοηθεί.
# 
# Για αύξηση των αριθμών των epochs επίσης δεν έχουμε καλύτερο αποτέλεσμα. Για να επηρεάσει ουσιαστικά ο αριθμός εποχών θα πρέπει να έχουμε ενα αρκετά μεγάλο σύνολο δεδομένων ώστε ο αλγόριθμος να μην κανει πρόωρα converge. Για εμάς και μικρός αριθμός εποχών ειναι αρκετος. 
# 
# Συμπερασματικά  ο πιο καθοριστικός παράγοντας για σωστή εξαγωγή αναπαραστάσεων ειναι ο αριθμός των δεδομένων εκπαίδευσης.
# 

# In[12]:


#Step 9(c (i))

def similarity(w,s,e):
    model = Word2Vec(final_list, window=w, size=s, workers=4)
    model.train(final_list, total_examples=len(final_list), epochs=e)
    voc = model.wv.index2word
    dim = model.vector_size
    rand_words = [voc[i] for i in idxs]
    for word in rand_words:
        print(f'Most similar words to "{word}":')
        for word,sim in model.wv.most_similar(word)[0:2]:
            print(f'     "{word}" -- sim: {sim}')
            


# In[33]:


similarity(10,100,2000)


# In[34]:


similarity(15,100,2000)


# In[31]:


#Step 9(c (ii))

similarity(5,100,2000)


# In[35]:


#Step 9(c (iii))

similarity(10,100,2000)


# In[32]:


similarity(5,100,3000)


# <h1 align = "center" >Μέρος 1 </h1>
# <h1 align = "center" >Ορθογράφος </h1>
# 
# Στο πρώτο μέρος επεκτείνουμε τον ορθογράφο που έχουμε ήδη υλοποιήσει χρησιμοποιώντας character level και word level unigram γλωσσικά μοντέλα, ενώ θα γίνει πειραματισμός και με bigram γλωσσικά μοντέλα. 

# ### Βήμα 10
# Για να βελτιώσουμε την απόδοση του ορθογράφου μας θα πρέπει να πετύχουμε την μέγιστη αξιοποίηση του συνόλου εκπαίδευσης. Για τον σκοπό αυτό εξάγουμε στατιστικά χαρακτηριστικά απο τα δεδομένα και ενσωματώνουμε την πληροφορία αυτή αλλάζοντας τα βάρη του μοντέλου. Οι πηγές στατιστικών θα είναι:
# * word level: εξάγουμε την πιθανότητα εμφάνισης κάθε λέξης
# * character level: εξάγουμε την πιθανότητα εμφάνισης κάθε χαρακτήρα
# 

# In[28]:


#Step 10
from collections import Counter

word_prob = Counter(text) # Counter returns a dictionary {word: freq} in a fast way
word_prob = {word:prob/len(text) for word,prob in word_prob.items()}

chars = [char for word in text for char in word]
char_prob = Counter(chars)
char_prob = {char:prob/len(chars) for char,prob in char_prob.items()}
max_prob_word = max(word_prob, key=word_prob.get)
min_prob_word = min(word_prob, key=word_prob.get)
print(f"The most probable word in the dictionary: '{max_prob_word}' with probability: {word_prob[max_prob_word]}")
print(f"Least probable word in the dictionary: '{min_prob_word}' with probability: {word_prob[min_prob_word]}")




# ### Βήμα 11
# Έχουμε ήδη δουλέψει με την απόσταση **Levenshtein** (ή edit distance). Χρησιμοποιούμε 3 τύπους απο edits: 
# * Εισαγωγή χαρακτήρα 
# * Διαγραφή χαρακτηρα 
# * Αντικατάσταση χαρακτήρα
# α) Υπολογίζουμε την μέση τιμή των βαρών του word level μοντέλου δηλαδή:
# 
# $$ W_{word}^{average} = \sum_{i} \cdot -log(~p(word_i)~) / |words|$$
# 
# Τα βάρη δίνονται απο την συνάρτηση $-log(p(x))$ , η οποία διαισθητικά κωδικοποιεί σωστά την πληροφορία, δίνοντας μεγαλύτερο βάρος στα λιγότερο συχνά ενδεχόμενα.
# Το κόστος των edits για το word-level μοντέλο είναι η μέση τιμή $w  = \overline{W} $. 
# 
# Εναλλακτικά μπορούμε να υπολογίσουμε την πιθανοτική μέση τιμή των βαρών αν θεωρήσουμε τυχαία μεταβλητή με τιμές τα βάρη και πιθανότητες την πιθανότητα της αντίστοιχης λέξης. Τότε η μέση τιμή των βαρών είναι επίσης η *εντροπία* της κατανομής των λέξεων $p(x), x \in word~corpus$. 
# 
# $$ \mathbb{E}[W_{word}] = -\sum_{i} p(word_i) \cdot log(~p(word_i)~) $$
# 
# (β) Κατασκευάζουμε έναν μετατροπέα με μία κατάσταση που υλοποιεί την απόσταση Levenshtein. Για κάθε edit το κόστος είναι $w$, εκτός απο την αντικατάσταση ενός γράμματος με τον εαυτό του που έχει κόστος 0. 
# 
# (γ) Επαναλαμβάνουμε για το unigram γλωσσικό μοντέλο.
# 
# (δ) Όπως έχουμε αναφέρει αυτός ο τρόπος υπολογισμού των βαρών δεν κωδικοποιεί σημαντική πληροφορία και δεν βελτιώνει την απόδοση του μοντέλου μας.
# 
# Ιδανικά θα θέλαμε ένα σύνολο labeled δεδομένων της μορφής (original word, wrong spelling). Με αυτό τον τρόπο μπορούμε να εξάγουμε σημαντική πληροφορία βρίσκοντας την πιθανότητα ενός συγκεκριμένου edit. 
# 
# Διαισθητικά δεν είναι όλες οι μετατροπές το ίδιο πιθανές. Για παράδειγμα μια λέξη που ξεκινά απο $a$ είναι σχεδόν αδύνατον να γραφεί λανθασμένα με $z$ (θεωρώντας ρεαλιστικά ανθρώπινα δεδομένα και όχι τυχαίο θωρυβώδες dataset). Θα θέλαμε λοιπόν να υπολογίσουμε την _a priori_ πιθανότητα κάθε edit και αυτήν να κωδικοποιήσουμε στα βάρη μας:
# 
# $$ Pr[ a\rightarrow b] = \frac{|error = b \rightarrow a|}{|train~samples|}, \forall a,b \in \{A+\epsilon\} $$, όπου $A$ το αλφάβητο. 
# 
# Το βάρος για μια συγκεκριμένη διόρθωση Θα πρέπει να είναι αντιστρόφως ανάλογο της πιθανότητας εμφανίσης του αντίστοιχου λάθους. Η ιδιότητα μπορεί να επιτευχθεί με την συνάρτηση  $ -log(p(X)) $ όπου $p(X)$ η συνάρτηση κατανομής πιθανοτήτων.
# 
# 

# In[29]:


#Step 11(a)
import math

# We create the weight dictionaries with an elegant dict comprehension..
weight_words = {word: -math.log(prob,2) for word,prob in word_prob.items()}
weight_chars = {char: -math.log(prob,2) for char,prob in char_prob.items()}

avg_weight_words = sum(list(weight_words.values()))/len(list(weight_words.values()))
print(f"Average weight for word-level model: {avg_weight_words}")

avg_weight_chars = sum(list(weight_chars.values()))/len(list(weight_chars.values()))
print(f"Average weight for unigram model: {avg_weight_chars}")


# In[30]:


#Step 11(b)

create_lev("lev_word.fst",alphabet,avg_weight_words)
create_lev("lev_unigram.fst",alphabet,avg_weight_chars)


# In[31]:


get_ipython().system('fstcompile --isymbols=chars.syms --osymbols=chars.syms  lev_word.fst lev_word.bin.fst')


# In[32]:


get_ipython().system('fstcompile --isymbols=chars.syms --osymbols=chars.syms  lev_unigram.fst lev_unigram.bin.fst')


# ### Βήμα 12
# Κατασκευάζουμε έναν αποδοχέα που αποδέχεται κάθε λέξη του corpus. Για βάρος χρησιμοποιούμε το $-log(P(word))$, δίνοντας στο μοντέλο περισσότερη πληροφορία και βελτιώνωντας το κριτήριο επιλογής λέξης. 
# 
# Ακολουθούμε την διαδικασία τόσο για το unigram όσο και για το word level γλωσσικό μοντέλο. 
# Έχουμε τροποποθήσει την συνάρτηση δημιουργίας αποδοχέα στο βήμα 6 έτσι ώστε να περιλαμβάνει περιπτώσεις για τους διαφορετικούς τύπους μοντέλων: 
# * Simple acceptor (zero weights)
# * Word level 
# * Unigram
# * Bigram
# 
# Στην συνέχεια βελτιστοποιούμε τα μοντέλα με τις _fstdeterminize, fstrmepsilon, fstminimize_.
# 

# In[33]:



create_acceptor(word_corpus,"acceptor_word.fst",weight_words,model='Word')
create_acceptor(word_corpus,"acceptor_unigram.fst",weight_chars,model='Unigram')


# In[34]:


get_ipython().system('fstcompile --isymbols=chars.syms --osymbols=chars.syms  acceptor_word.fst acceptor_word.bin.fst')

get_ipython().system('fstdeterminize acceptor_word.bin.fst acceptor_word.bin.fst')
get_ipython().system('fstrmepsilon acceptor_word.bin.fst acceptor_word.bin.fst')
get_ipython().system('fstminimize acceptor_word.bin.fst acceptor_word.bin.fst')


# In[35]:


get_ipython().system('fstcompile --isymbols=chars.syms --osymbols=chars.syms  acceptor_unigram.fst acceptor_unigram.bin.fst')

get_ipython().system('fstdeterminize acceptor_unigram.bin.fst acceptor_unigram.bin.fst')
get_ipython().system('fstrmepsilon acceptor_unigram.bin.fst acceptor_unigram.bin.fst')
get_ipython().system('fstminimize acceptor_unigram.bin.fst acceptor_unigram.bin.fst')


# ### Βήμα 13
# Ακολουθώντας την διαδικασία του ερωτήματος 7 θα κατασκευάσουμε ορθογράφο με το word-level γλωσσικό μοντέλο και μετατροπέα, και αντίστοιχα με το unigram μοντέλο και word-level μετατροπέα.
# 
# Περιμένουμε οι ορθογράφοι αυτοί να έχουν καλύτερη απόδοση απο τον απλοϊκό ορθογράφο του βήματος 7, καθώς κωδικοποιούν πληροφορία και στα βάρη τους. 
# 
# Σημειώνουμε οτι σε όλα τα μοντέλα χρησιμοποιούμε τον word-level μετατροπέα για τον υπολογισμό της Levenshtein απόστασης, όπως δηλώνεται στην εκφώνηση. Εναλλακτικά για χρήση του αντίστοιχου μετατροπέα με το αντίστοιχο μοντέλο είδαμε ότι έχουμε μειωμένη απόδοση. Το αποτέλεσμα είναι λογικό γιατί όπως δείχνουμε παρακάτω το μοντέλο λειτουργεί καλύτερα με αναπαραστάσεις σε επίπεδο λέξεων.

# In[36]:


#Step 13(a)

get_ipython().system('fstarcsort --sort_type=olabel lev_word.bin.fst lev_word.bin.fst')
get_ipython().system('fstcompose  lev_word.bin.fst acceptor_word.bin.fst spell_checker_word.bin.fst')


# In[37]:


#Step 13(b)


get_ipython().system('fstcompose  lev_word.bin.fst acceptor_unigram.bin.fst spell_checker_unigram.bin.fst')
#!fstarcsort --sort_type=olabel lev_unigram.bin.fst lev_unigram.bin.fst
#!fstcompose  lev_unigram.bin.fst acceptor_unigram.bin.fst spell_checker_unigram.bin.fst


# In[38]:


#Step 13(c)
# We repeat the process in step 7.. 


word = ['cit']
create_acceptor(word,"in.fst")

# Word-level model
get_ipython().system('fstcompile  --isymbols=chars.syms --osymbols=chars.syms in.fst in.bin.fst')
get_ipython().system('fstarcsort --sort_type=ilabel spell_checker_word.bin.fst spell_checker_word.bin.fst ')
get_ipython().system('fstarcsort --sort_type=olabel in.bin.fst in.bin.fst ')

print(f"Min distance prediction for {word} is: ")
get_ipython().system('fstcompose in.bin.fst spell_checker_word.bin.fst |fstshortestpath --nshortest=1 | fstrmepsilon |  fsttopsort |fstprint -isymbols=chars.syms  -osymbols=chars.syms| cut -f4 | grep -v "<epsilon>" |head -n -1 | tr -d \'\\n\'')


# In[39]:


# Unigram model
get_ipython().system('fstcompile  --isymbols=chars.syms --osymbols=chars.syms   in.fst in.bin.fst')
get_ipython().system('fstarcsort --sort_type=ilabel spell_checker_unigram.bin.fst spell_checker_unigram.bin.fst ')
get_ipython().system('fstarcsort --sort_type=olabel in.bin.fst in.bin.fst ')

print(f"Min distance prediction for {word} is: ")
get_ipython().system('fstcompose in.bin.fst spell_checker_unigram.bin.fst |fstshortestpath --nshortest=1 | fstrmepsilon |  fsttopsort |fstprint -isymbols=chars.syms  -osymbols=chars.syms| cut -f4 | grep -v "<epsilon>" |head -n -1 | tr -d \'\\n\'')


# (γ) Οι δύο ορθογράφοι που δημιουργήσαμε έχουν ίδια αρχή λειτουργίας και αντιμετωπίζουν παρόμοια προβλήματα. Συγκεκριμένα η εκτίμηση τους είναι η λέξη με την ελάχιστη Levenhstein απόσταση απο την λέξη εισόδου. Η διαφορά είναι οτι στο word-level μοντέλο η επιλογή σταθμίζεται απο το βάρος της λέξης που τελικά επιλέγουμε, ενώ στο unigram μοντέλο σταθμίζεται αντιστοιχα κάθε επιλογή χαρακτήρα.
# 
# Το μοντέλο επηρεάζεται σημαντικά απο το περιορισμένο λεξιλόγιο του, έτσι αν δεν γνωρίζει την ύπαρξη μίας λέξης δεν μπορεί να διορθώσει σε αυτή. Αυτός είναι ένας λογικός περιορισμός και διορθώνεται με την αύξηση των train δεδομένων. 
# 
# Το δεύτερο σημαντικό ελάττωμα του μοντέλου είναι οτι δεν διαθέτει αρκετά καλο κριτήριο για την επιλογή λέξεων σε περίπτωση ισοπαλίας. Αυτό έχει σε έναν βαθμό διορθωθεί με την χρήση στατιστικών στοιχείων στα παραπάνω 2 μοντέλα. 
# 
# Η αμφισημία προκύπτει γιατί για μία λέξη όπως το 'cit' υπάρχουν πολλές πιθανές λέξεις διόρθωσης, όπως έχουμε ήδη δέιξει. Παρ' όλα αυτά και τα δύο μοντέλα (word-level, unigram) κρίνουν ώς την πιο πιθανή λέξη την λέξη it. 
# 

# ### Βήμα 14
# Θα αξιολογήσουμε τους δύο ορθογράφους που δημιουργήσαμε πάνω στο spell checker test set που είχαμε δεί στο βήμα 8. Θα χρησιμοποιήσουμε την συνάρτηση που ήδη έχουμε γράψει στο βήμα 8. 
# 
# Χωρίζουμε το σύνολο δεδομένων αξιολόγησης σε _X_test, y_test_ που αποτελούν _270_  ζεύγη της μορφής: (correct spelling, wrong spelling). Πάνω σε αυτα υπολογίζουμε το accuracy για κάθε μοντέλο.
# 

# In[40]:


#Too slow, need help :'( 
predict(y_test,X_test,"spell_checker.bin.fst",Show = False)
predict(y_test,X_test,"spell_checker_word.bin.fst",Show = False)
predict(y_test,X_test,"spell_checker_unigram.bin.fst",Show = False)


# **Spell checker Simple**: Το αρχικό μοντέλο δεν περιέχει καμία πληροφορία στα βάρη του, όμως έχει αρκετά καλη απόδοση **0.592 %**. Το μοντέλο μπορεί να προβλέψει με ακρίβεια σπάνιες λέξεις που ήδη γνωρίζει, ή μεγάλες σε μήκος λέξεις που είναι δύσκολο να μπερδευτούν με άλλες. 
# 
# **Spell checker Word Level**: Το μοντέλο με βάρη βασισμένα στην πιθανότητα εμφάνισης λέξεων έχει την καλύτερη απόδοση **0.618 %**. Απο αυτό συμπεραίνουμε οτι αναπαραστάσεις σε επίπεδο λέξεων κωδικοποιούν καλύτερα την πληροφορία. Συγκεκριμένα ειναι καλύτερα το μοντέλο να ψάχνει ένα token-λέξη που έχει την ελάχιστη απόσταση με βάση την a priori γνώση του, παρά να προσπαθεί κάθε φορά να μαντέψει τον πιο πιθανό χαρακτήρα. 
# 
# **Spell Checker Unigram**: Η unigram αναπαράσταση είχε την μικρότερη ακρίβεια **0.537**, και όπως είπαμε τα βάρη ανα χαρακτήρα αποπροσανατολίζουν το μοντέλο και δεν επιλέγει σωστά. Σημειώνουμε οτι για unigram μετατροπέα το μοντέλο είχε ακρίβεια ~32%.
# 
# 
# 

# ### Βήμα 15
# Θα εκτελέσουμε τα προηγούμενα βήματα για ενα Bigram γλωσσικό μοντέλο.
# Το bigram γλωσσικό μοντέλο ανήκει στην ευρύτερη κλάση των n-gram μοντέλων και προκύπτει για n = 2. Με τον όρο n-gram αναφερόμαστε σε μία συνεχή ακολουθία n αντικειμένων απο ένα δείγμα φωνής ή κειμένου Συγκεκριμένα στο πλάισιο των λέξεων ένα n-gram γλωσσικό μοντέλο χρησιμοποιεί ακολουθίες n χαρακτήρων για να προβλέψει το επόμενο γράμμα. Η πρόβλεψη βασίζεται σε (n-1)-order αλυσίδα markov (δηλαδή ισχύει η μαρκοβιανή ιδιότητα αλλα η εξάρτηση σταματάει στα n προηγούμενα δείγματα). 
# 
# $$ \mathbb{P}[x_i|x_{i-1},x_{i-2},...,x_0] = \mathbb{P}[x_i|x_{i-1},x_{i-2},...,x_{i-(n-1)}] $$ 
# 
# Δυο πλεονεκτήματα τους ειναι:
# * Η απλότητα
# * Η κλιμακωσιμότητα
# 
# Για το bigram μοντέλο αρκεί να υπολογίσουμε τις πιθανότητες:
# 
# $$ \mathbb{P}[x_i|x_{i-1}] $$
# για κάθε ζεύγος στο αλφάβητο μας (συν το $\epsilon$).
# 
# 
# Στο παρακάτω κελί υπολογίζουμε με παρόμοιο τρόπο τις πιθανότητες εμφάνισης για κάθε bigram. Συγκεκριμένα έχουμε επαυξήσει κάθε λέξη του συνόλου ώστε να αρχίζει απο το κενό, αυτή είναι η πιθανότητα να επιλεγεί το γράμμα απο το οποίο αρχίζει η λέξη χωρίς να έχει προηγηθεί κάποιο άλλο. Έπειτα υπολογίζουμε τα κατάλληλα βάρη και υλοποιούμε τον μετατροπέα και τον αποδοχέα για να τους συνδυάσουμε στον bigram spell checker. 
# 

# In[41]:


extended_text = [ " " + word for word in text] # " " symbolizes epsilon, for bigrams like (<epsilon>,a)"
bigrams = [(char1,char2) for word in extended_text for char1,char2 in zip(word,word[1:])]
bigram_prob = Counter(bigrams)
bigram_prob = {bigram:prob/len(bigrams) for bigram,prob in bigram_prob.items()}

weight_bigrams =  {bigram: -math.log(prob,2) for bigram,prob in bigram_prob.items()}
avg_weight_bigrams = sum(list(weight_bigrams.values()))/len(list(weight_bigrams.values()))


# In[42]:


create_lev("lev_bigram.fst",alphabet,avg_weight_bigrams)
get_ipython().system('fstcompile --isymbols=chars.syms --osymbols=chars.syms  lev_bigram.fst lev_bigram.bin.fst')

create_acceptor(word_corpus,"acceptor_bigram.fst",weight_bigrams,model='Bigram')

get_ipython().system('fstcompile --isymbols=chars.syms --osymbols=chars.syms  acceptor_bigram.fst acceptor_bigram.bin.fst')

get_ipython().system('fstdeterminize acceptor_bigram.bin.fst acceptor_bigram.bin.fst')
get_ipython().system('fstrmepsilon acceptor_bigram.bin.fst acceptor_bigram.bin.fst')
get_ipython().system('fstminimize acceptor_bigram.bin.fst acceptor_bigram.bin.fst')


get_ipython().system('fstarcsort --sort_type=olabel lev_word.bin.fst lev_word.bin.fst')
#!fstarcsort --sort_type=olabel lev_bigram.bin.fst lev_bigram.bin.fst

get_ipython().system('fstcompose  lev_word.bin.fst acceptor_bigram.bin.fst spell_checker_bigram.bin.fst')
#!fstcompose  lev_bigram.bin.fst acceptor_bigram.bin.fst spell_checker_bigram.bin.fst

get_ipython().system('fstdeterminize acceptor_bigram.bin.fst acceptor_bigram.bin.fst')
get_ipython().system('fstrmepsilon acceptor_bigram.bin.fst acceptor_bigram.bin.fst')
get_ipython().system('fstminimize acceptor_bigram.bin.fst acceptor_bigram.bin.fst')


predict(y_test,X_test,"spell_checker_bigram.bin.fst",Show = False)


# Παρατηρούμε οτι με το bigram μοντέλο η ακρίβεια πάνω στο test set έπεσε στο **_0.381%_**. Αυτο επιβεβαιώνει την αντίληψη μας ότι μοντέλα βασισμένα σε χαρακτήρες δεν λειτουργούν καλά για το συγκεκριμένο πρόβλημα, και μία προσέγγιση με στοιχεία λέξεις θα έχει πολυ καλύτερα αποτελέσματα.
# 
# Παρακάτω ελέγχουμε τους ορθογράφους σε τυχαίο σύνολο δεδομένων:

# In[43]:


idxs = random.sample(range(0, len(y_test)), 20)
X_rand = [X_test[i] for i in idxs]
Y_rand = [y_test[i] for i in idxs]

predict(Y_rand,X_rand,"spell_checker.bin.fst",Show = False)
predict(Y_rand,X_rand,"spell_checker_word.bin.fst",Show = False)
predict(Y_rand,X_rand,"spell_checker_unigram.bin.fst",Show = False)
predict(Y_rand,X_rand,"spell_checker_bigram.bin.fst",Show = False)


# <h1 align = "center">ΜΕΡΟΣ 2</h1>
# 
# <h3 align = "center">Χρήση σημασιολογικών αναπαραστάσεων για ανάλυση συναισθήματος.</h3>
# 
# Στο δεύτερο αυτό μέρος θα αξιολογήσουμε διάφορα μοντέλα και λεκτικές αναπαραστάσεις (embeddings) για το πρόβλημα της αναγνώρισης συναισθήματος. Θα χρησιμοποιήσουμε την γνωστή βάση δεδομένων απο reviews της _IMDB_, και θα ταξινομήσουμε τις κριτικές σε θετικές και αρνητικές με βάση το συναίσθημα. 

# ### Βήμα 16
# Κατεβάζουμε τα δεδομένα κριτικών απο το Stanford. Για την επεξεργασία τους χρησιμοποιούμε τον προτινώμενο κώδικα. 

# In[44]:


#Step 16(a)

#!wget -P ./data/ http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz


# In[45]:


#Step 16(b)

import os

data_dir = './data/aclImdb_v1/aclImdb/'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
pos_train_dir = os.path.join(train_dir, 'pos')
neg_train_dir = os.path.join(train_dir, 'neg')
pos_test_dir = os.path.join(test_dir, 'pos')
neg_test_dir = os.path.join(test_dir, 'neg')

# For memory limitations. These parameters fit in 8GB of RAM.
# If you have 16G of RAM you can experiment with the full dataset / W2V
MAX_NUM_SAMPLES = 5000
# Load first 1M word embeddings. This works because GoogleNews are roughly
# sorted from most frequent to least frequent.
# It may yield much worse results for other embeddings corpora
NUM_W2V_TO_LOAD = 1000000

import numpy as np

SEED = 42

# Fix numpy random seed for reproducibility
np.random.seed(42)

try:
    import glob2 as glob
except ImportError:
    import glob

import re

def strip_punctuation(s):
    return re.sub(r'[^a-zA-Z\s]', ' ', s)

def preprocess(s):
    return re.sub('\s+',' ', strip_punctuation(s).lower())

def tokenize(s):
    return s.split(' ')

def preproc_tok(s):
    return tokenize(preprocess(s))

def read_samples(folder, preprocess=lambda x: x):
    samples = glob.iglob(os.path.join(folder, '*.txt'))
    data = []
    for i, sample in enumerate(samples):
        if MAX_NUM_SAMPLES > 0 and i == MAX_NUM_SAMPLES:
            break
        with open(sample, 'r') as fd:
            x = [preprocess(l) for l in fd][0]
            data.append(x)
    return data

def create_corpus(pos, neg):
    corpus = np.array(pos + neg)
    y = np.array([1 for _ in pos] + [0 for _ in neg])
    indices = np.arange(y.shape[0])
    np.random.shuffle(indices)
    return list(corpus[indices]), list(y[indices])


# In[46]:


#Step 16(b)

(X_train,y_train) = create_corpus(read_samples(pos_train_dir),read_samples(neg_train_dir))
(X_test,y_test) = create_corpus(read_samples(pos_test_dir),read_samples(neg_test_dir))


# ### Βήμα 17
# 
# 
# Μια απλοϊκή αναπαράσταση για μία πρόταση ειναι η _Bag of Words_. Με βάση αυτή κάθε λέξη κωδικοποιείται σαν ένα _one hot encoding_ πάνω στο λεξιλόγειο. Δηλαδή ένα διάνυσμα (μεγέθους ίσου με το λεξικό) με '1' στην θέση που αντιστοιχεί στην λέξη και '0' σε όλες τις άλλες.  Η αναπάσταση της πρότασης ειναι απλά το διανυσματικό άθροισμα των one hot encodings των λέξεων της. Μία τέτοια προσέγγιση είναι μη αποδοτική καθώς ειναι ιδιαίτερα _sparse_, απαιτώντας πολυ μνήμη χωρίς να έχουμε κάποιο πλεονέκτημα (πέραν της απλότητας). 
# 
# Μία βελτίωση του _BoW_ μοντέλου είναι το σταθμισμένο άθροισμα των one-hot vectors. Κάτι τέτοιο μπορεί να γίνει με βάρη **TF-IDF**
# 
# Το TF-IDF αποτελείται από 2 όρους. Ο πρώτος είναι το **Term Frequency (TF)**:
# 
# $$ tf(i,d) = \frac{f(i,d)}{\sum_{i} f(i,d)}$$
# 
# Όπου *i* ο όρος στο κείμενο *d*. Το tf είναι στην ουσία η συχνότητα με την οποία εμφανίζεται ο κάθε όρος στο κείμενο. Λέξεις με μεγάλη συχνότητα είναι σημαντικότερες για το κείμενο από ότι οι λέξεις με μικρή συχνότητα.
# 
# Ο δεύτερος όρος στο TF-IDF είναι το **Inverse Document Frequency**:
# 
# $$ idf(i) = log \frac{N}{df(i)}$$
# 
# Όπου *Ν* ο αριθμός των κειμένων και *df(i)* ο αριθμός των κειμένων στους οποίους εμφανίζεται ο όρος *i*. Το idf είναι ένας δείκτης της πληροφορίας που δίνει η κάθε λέξη. Αν η λέξη εμφανίζεται σε όλα τα κείμενα τότε αυτή δε δίνει καθόλου πληροφορία και το κλάσμα θα λάβει τιμή 1, και απο τον λογάριθμ θα μετατραπεί στην τιμή 0. Σε όσο πιο λίγα κείμενα εμφανίζεται η λέξη τόσο περισσότερη πληροφορία έχει, και τόσο πιο σημαντική είναι η εμφάνιση της. Αυτό αντιστοιχεί γενικά στην έννοια της πληροφορίας κατα Shannon, και συγκεκριμένα τα _TF-IDF_ συνδέονται με την απο κοινού πληροφορία των κειμένων. 
# 
# Υπάρχουν διάφορες παραλλαγές για τον υπολογισμό των βαρών  εμείς χρησιμοποιούμε την πιο απλή. 
# 
# 
# Το TF-IDF υπολογίζεται τελικά ως το γινόμενο των δύο όρων:
# 
# $$ tfidf(i) = tf(i,d) \cdot idf(i)$$
# 
# Άρα, αν το γινόμενο TF-IDF είναι υψηλό, τότε η λέξη i είναι σημαντική πληρορορία στο κείμενο d αφού η λέξη αυτή εμφανίζεται πολλές φορές στο κείμενο και δεν εμφανίζεται σε πολλά από τα Ν κείμενα που που εξετάζονται.

# ***Βήμα 17 (β, γ, δ)***
# 
# Σε αυτά τα βήματα θα χρησιμοποιήσουμε τον transformer CountVectorizer του sklearn για την εξαγωγή μη σταθμισμένων BOW αναπαραστάσεων και θα εκπαιδεύσουμε τον ταξινομητή LogisticRegression για να ταξινομήσουμε τα σχόλια σε θετικά και αρνητικά. 
# 
# O **CountVectorizer** μετατρέπει μία συλλογή από κείμενα σε έναν πίνακα στον οποίο αποθηκεύεται ο αριθμός εμφάνισης των tokens. Αυτή η υλοποιήση παράγει μία αραιή αναπαράσταση του αριθμού εφάνισης των tokens χρησιμοποιώντας το scipy.sparse.csr_matrix.
# 
# Ο **TfidfVectorizer** μετατρέπει μια συλλογή από μη επεξεργασμένα κείμενα σε έναν πίνακα με τις TF-IDF τιμές των tokens. Είναι ισοδύναμος με τον CountVectorizer αλλά χρησιμοποιεί έναν TfidfTransformer. Ο TfidfTransformer υπολογίζει τα γινόμενα TF-IDF για κάθε token του κειμένου.
# 
# Ο **Logistic Regression** ταξινομητής χρησιμοποιείται για να μοντελοποιήσουμε την πιθανότητα να επιλεγεί μια συγκεκριμένη κλάση ή να συμβεί ένα γεγονός. Μπορεί να χρησιμοποιήθεί για την κατηγοριοποίηση κλάσεων σε δύο κατηγορίες όπως θετικές και αρνητικές αρνητικές κριτικές, είτε για την κατηγοριοποίηση κλάσεων σε περισσότερες από δύο κατηγορίες. Μπορεί να θεωρηθεί μια ειδική μέθοδος γραμμικής παλινδρομησης, όπου προβλέπουμε την πιθανότητα κάνωντας fit σε μια logistic συνάρτηση (δηλαδή ενα σιγμοειδές)
# 
# ![logistic-regression](http://juangabrielgomila.com/wp-content/uploads/2015/04/LogReg_1.png)
# 
# 

# In[47]:


#Step 17(b)

#import libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


# In[48]:


#Step 17(b,c)
#Count Vectorizer

vectorizer = CountVectorizer()
# X_train_BOW_C[i] is the bag of words CountVectorizer representation for the i-th comment
X_train_BOW_C = vectorizer.fit_transform(X_train)
vectorizer_test_C = CountVectorizer(vocabulary = vectorizer.get_feature_names())
X_test_BOW_C = vectorizer_test_C.fit_transform(X_test)
LG_C = LogisticRegression(random_state=0, multi_class = 'ovr', solver = 'liblinear',penalty = 'l2')
xx_C = LG_C.fit(X_train_BOW_C,y_train)


# In[49]:


#Step 17(d)

vectorizer = TfidfVectorizer()
# X_train_BOW_T[i] is the bag of words TfidfVectorizer representation for the i-th comment
X_train_BOW_T = vectorizer.fit_transform(X_train)
vectorizer_test_T = TfidfVectorizer(vocabulary = vectorizer.get_feature_names())
X_test_BOW_T = vectorizer_test_T.fit_transform(X_test)
LG_T = LogisticRegression(random_state=0, multi_class = 'ovr', solver = 'liblinear',penalty = 'l2')
xx_T = LG_T.fit(X_train_BOW_T,y_train)


# ***Σύγκριση αποτελεσμάτων*** 
# 
# Συγκρίνοντας τα αποτελέσματα των δύο Vectorizer() παρατηρούμε ότι τα δύο ποσοστά έχουν πολύ μιρκή διαφορά. Καλύτερη ακρίβεια έχει ο TfidfVectorizer. Αυτό συμβαίνει γιατί η χρήση tf-idf συχνοτήτων συμβάλλει στην μείωση της επίδρασης των tokens που εμφανίζονται πολύ συχνά στο κείμενο. Όπως είπαμε και παραπάνω, τα tokens αυτά δεν δίνουν τόση πληροφορία για το κείμενο όσο οι λέξεις που εμφανίζονται λιγότερο και για αυτό είναι καλύτερα να μην λαμβάνονται υπόψιν κατά τη διαδικασία του classification. 
# 

# In[50]:


#Step 17(d)

acc_vec = 0
for sample,label in zip(X_test_BOW_C,y_test):
    if xx_C.predict(sample) == label:
        acc_vec+=1
print(f'The accuracy of the Count Vectorizer is: {acc_vec/len(X_test)}')

acc_tfidf = 0
for sample,label in zip(X_test_BOW_T,y_test):
    if xx_T.predict(sample) == label:
        acc_tfidf+=1
print(f'The accuracy of the Tfidf Vectorizer is: {acc_tfidf/len(X_test)}')


# ### Βήμα18 
# 
# Οι λέξεις μπορούν να αναπαρασταθούν και απο προεκπαιδευμένα embeddings. Όπως τα word2vec embeddings που έχουμε δει σε προηγούμενα ερωτήματα .Αυτά τα embeddings προκύπτουν από ένα νευρωνικό δίκτυο με ένα layer. Υπάρχουν δύο διαφορετικές προσεγγίσεις με παρόμοια αποτελέσματα. 
# 
# Στην πρώτη το νευρωνικό δίκτυο το οποίο καλείται να προβλέψει μια λέξη με βάση το context της, ένα κυλιόμενο παράθυρο το μέγεθος του οποίο αποτελεί παράμετρο του μοντέλου. Αυτο αποτελεί το **CBOW** μοντέλο. Στην δεύτερη το δίκτυο καλείται να προβλέψει το context με βάση τη λέξη . Και αυτο αποτελεί το **Skip-gram** μοντέλο. 
# 
# Τα word2vec vectors είναι πυκνές (dense) αναπαραστάσεις σε λιγότερες διαστάσεις από τις BOW και κωδικοποιούν σημασιολογικά χαρακτηριστικά μιας λέξης με βάση την υπόθεση ότι λέξεις με παρόμοιο νόημα εμφανίζονται σε παρόμοιες θέσεις στο κείμενο. Μια πρόταση μπορεί να αναπαρασταθεί ως ο μέσος όρος των w2v διανυσμάτων κάθε λέξης που περιέχει, για λέξεις που δνε υπάρχουν προσθέτουμε το μηδενικό διάνυσμα, η τεχνική αυτή ονομάζεται **Neural Bag of Words**.

# 
# 
# Σε αυτό το βήμα υπολογίσαμε το ποσοστό των out of vocabulary (OOV) words για τις αναπαραστάσεις που υπολογίσαμε στο Βήμα 9. Το ποσοστό που ζητείται δίνεται από τον τύπο 
# $$ OOV =\frac{\text{num of unique words in X_train - num of words in  voc of word2vec}}{\text{num of unique words in X_train}}$$ 

# In[51]:


#Step 18(a)
#OOV words

critics_words = []
for critic in X_train:
    critics_words += nltk.word_tokenize(critic)

critics_corpus = set(critics_words)
w2vec_corpus = set(voc)
OOV = critics_corpus.difference(w2vec_corpus)
# We compute out of Voc words as a set difference
    
    
print(f'The percentage of OOV is: {100*float(len(OOV))/len(critics_corpus)} %')


# (β)
# Σε αυτό το βήμα θα χρησιμοποιήσουμε τα embeddings που προκύπτουν από το word_corpus που δημιουργήσαμε από τα τρία βιβλία που αναφέρονται στην αρχή. Στη συνέχεια, θα χρησιμοποιήσουμε αυτά τα embeddings για την κατασκευή Neural Bag of Words αναπαραστάσεων για κάθε σχόλιο στο corpus(κριτικές από το IMDB) και θα εκπαιδεύσουμε ένα Logistic Regression μοντέλο για ταξινόμηση των κριτικών σε θετικές και αρνητικές. 

# In[52]:


#Step 18(b)

import numpy as np

# Convert to numpy 2d array (n_vocab x vector_size)
def to_embeddings_Matrix(model):  
    embedding_matrix = np.zeros((len(model.wv.vocab), model.vector_size))
    word2idx = {}
    for i in range(len(model.wv.vocab)):
        embedding_matrix[i] = model.wv[model.wv.index2word[i]]
        word2idx[model.wv.index2word[i]] = i
    return embedding_matrix, model.wv.index2word, word2idx

(embedding_matrix, model.wv.index2word, word2idx) = to_embeddings_Matrix(model)


# In[53]:


#Step 18(b)
#Neural Bag of Words for movie critics

#for every critic in test and train sets we create an Neural bag of words representation
#by adding the embeddings for the words in the critic and dividing by the length of it
    
train = np.zeros((len(X_train),model.vector_size))
count_c = 0                      
for critic in X_train:
    critic = nltk.word_tokenize(critic)
    for word in critic:
        if word in word2idx.keys():
            i = word2idx[word]
            train[count_c] += embedding_matrix[i]
    train[count_c] /= len(critic)
    count_c += 1
                    

test = np.zeros((len(X_test),model.vector_size))   
count_c = 0      
for critic in X_test:
    critic = nltk.word_tokenize(critic)
    for word in critic:
        if word in word2idx.keys():
            i = word2idx[word]
            test[count_c] += embedding_matrix[i]
    test[count_c] /= len(critic)
    count_c += 1


# In[54]:


#Step 18(b)

LG = LogisticRegression(random_state=0, multi_class = 'ovr', solver = 'liblinear',penalty = 'l2')
xx = LG.fit(train,y_train)
xx.predict(test)
print(f'The accuracy of the LogisticRegression Model with our word_corpus is: {xx.score(test,y_test)}')


# ***Συμπεράσματα***
# 
# Παρατηρούμε ότι το accuracy του μοντέλου μας είναι αρκετά χαμηλό. Αυτό συμβαίνει γιατί το ποσοστό των OOV είναι υψηλό και άρα πολλές λέξεις που υπάρχουν στις προτάσεις της κάθε κριτικής δεν συμβάλλουν στον υπολογισμό της Neural Bag of Words αναπαράστασης. Συμπεραίνουμε λοιπόν, ότι όσο μεγαλύτερο word_corpus έχουμε για την εκπαίδευση του μοντέλου μας και τη δημιουργία των embeddings, τόσο μεγαλύτερο accuracy θα έχουμε.

# In[55]:


#Step 18(c)

#Download pretrained GoogleNews vectors
#from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit


# (δ) 
# 
# Σε αυτό το βήμα φορτώσαμε τα GoogleNews με τη βιβλιοθήκη gensim και εξάγαμε αναπαραστάσεις (word2vec embeddings) με βάση αυτά. Στη συνέχεια, για τις λέξεις που είχαμε βρει το similarity με άλλες λέξεις στο corpus στο βήμα 9γ, υπολογίσαμε το similarity με βάση το μοντέλο με τα GoogleNews.

# In[17]:


#Step 18(d)

from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin',binary=True, limit=1000000)


# In[18]:


#Step 18(d)

#Similarity
#pick 10 random words from the dictionary
for word in rand_words:
    print(f'Most similar words to "{word}":')
    for word,sim in model.wv.most_similar(word):
        print(f'     "{word}" -- sim: {sim}')


# ***Συμπεράσματα***
# 
# Παρατηρούμε ότι το similarity για τις ίδιες λέξεις του ερωτήματος 9γ έχει πλέον αυξηθεί. Οι λέξεις που βρίσκει το μοντέλο με τα GoogleNews είναι πιο σχετικές με την υπο εξέταση λέξη, καθώς και το ποσοστό ομοιότητας έχει σχεδόν διπλασιαστεί. Αυτό είναι ένα αποτέλεσμα που περιμέναμε, από τη στιγμή που έχουμε ένα word_corpus κατά τάξεις μεγαλύτερο από αυτό που είχαμε δημιουργήσει στα πρώτα ερωτήματα. 

# (ε)
# 
# Σε αυτό το βήμα δημιουργήσαμε αναπαραστάσεις Neural Bag of Words για κάθε κριτική με τη χρήση των embeddings για το μοντέλο με τα Google News. Χρησιμοποιώντας αυτές τις αναπαραστάσεις εκπαιδεύσαμε ένα Logistic Regression Model για να αναγνωρίζει αν μία κριτική είναι θετική ή αρνητική.
# 
# Συγκρίνοντας το αποτέλεσμα του Logistic Regression Model που χρησιμοποιεί αναπαραστάσεις με τη χρησή των Google News, με το μοντέλο που χρησιμοποιεί αναπαραστάσεις με τη χρήση του word_corpus που δημιουργήσαμε, συμπεραίνουμε ότι το πρώτο έχει μεγαλύτερο accuracy. Αυτό συμβαίνει γιατί τα Google News, λόγω του μεγαλύτερου μεγέθους τους, δίνουν embeddings για περισσότερες λέξεις. Αρα η Neural Bag of Words αναπαράσταση για κάθε κριτική, δίνει καλύτερη πληροφορία για το αν είναι θετική ή αρνητικη και έτσι αυξάνεται η πιθανότητα να γίνει η ταξινόμηση της στη σωστή κατηγορία.

# In[66]:


#Step 18(e)
#Neural Bag of Words for movie critics
    
train_google = np.zeros((len(X_train),model.vector_size))
count_c = 0    
                      
for critic in X_train:
    critic = nltk.word_tokenize(critic)
    for word in critic:
        if word in model:
            train_google[count_c] += model[word]
    train_google[count_c] /= len(critic)
    count_c += 1
    
test_google = np.zeros((len(X_test),model.vector_size))   
count_c = 0
      
for critic in X_test:
    critic = nltk.word_tokenize(critic)
    for word in critic:
        if word in model:
            test_google[count_c] += model[word]
    test_google[count_c] /= len(critic)
    count_c += 1


# In[67]:


#Step 18(e)

LG = LogisticRegression(random_state=0, multi_class = 'ovr', solver = 'liblinear',penalty = 'l2')
xx = LG.fit(train_google,y_train)
xx.predict(test_google)
print(f'The accuracy of the LogisticRegression Model with GoogleNews is: {xx.score(test_google,y_test)}')


# ***Βήμα 18(στ)***
# 
# Σε αυτό το βήμα δημιουργείσαμε αναπαραστάσεις προτάσεων με χρήση σταθμισμένου μέσου των w2v αναπαραστάσεων των λέξεων. Ως βάρη χρησιμοποιήσαμε τα TF-IDF βάρη των λέξεων. 

# In[68]:


#Step 18(f)

tfidf = TfidfVectorizer(analyzer = nltk.word_tokenize)
X_train_tfidf = tfidf.fit_transform(X_train)
voc_train = tfidf.vocabulary_
X_test_tfidf = tfidf.fit_transform(X_test)
voc_test = tfidf.vocabulary_


# In[81]:


#Step 18(f)
#Neural Bag of Words for movie critics
    
train_tfidf = np.zeros((len(X_train),model.vector_size))
count_c = 0    
                      
for critic in X_train:
    critic = nltk.word_tokenize(critic)
    for word in critic:
        if word in model and word in voc_train:
            train_tfidf[count_c] += model[word]*X_train_tfidf[count_c,voc_train[word]]
    count_c += 1
                 

test_tfidf = np.zeros((len(X_test),model.vector_size))   
count_c = 0
count = 0
                 
for critic in X_test:
    critic = nltk.word_tokenize(critic)
    for word in critic:
        if word in model and word in voc_test:
            test_tfidf[count_c] += model[word]*X_test_tfidf[count_c,voc_test[word]]
    count_c += 1


# In[82]:


#Step 18(g)

LG = LogisticRegression(random_state=0, multi_class = 'ovr', solver = 'liblinear',penalty = 'l2')
xx = LG.fit(train_tfidf,y_train)
print(f'The accuracy of the LogisticRegression Model with GoogleNews tf-idf embeddings is: {xx.score(test_tfidf,y_test)}')


# ***Συμπεράσματα***
# 
# Τρέχοντας το LogisticRegression Model με τα GoogleNews δεδομένα για τα tf-idf embeddings, παρατηρούμε ότι το ποσοστό μειώνεται κατά 2% σε σχέση με τη χρήση embeddings χωρίς τα tf-idf βάρη. Αυτό συμβαίνει γιατί τα tf-idf μειώνουν την επίδραση των λέξεων που εμφανίζονται πολλές φορές στο κείμενο, καθώς και αυτών που εμφανίζονται πολύ λίγες φορές. Άρα, ενδέχεται στις λέξεις που δεν λήφθηκαν υπόψιν, να ήταν και κάποιες που έδιναν κάποια επιπλέον πληροφορία για το αν η κριτική ήταν θετική ή αρνητική.

# ### Βήμα 19
# 
# Σε αυτό το βήμα επιλέξαμε να συγκρίνουμε την επίδοση των KNN και SVM classifiers. 
# 
# **KNN classifier** :Ο classifier υποθέτει ότι οι αναπαραστάσεις των λέξεων με παρόμοια σημασία θα είναι και κοντά στον χώρο. Θεωρούμε λοιπόν, ότι για να κάνουμε classify, θα βρούμε τις 3 κοντινότερες αναπαραστάσεις στην αναπαράσταση της κριτικής και με βάση αυτές θα αποφανθούμε αν είναι θετική ή αρνητική.
# 
# **SVM classifiers** :Ο classifier αναπαριστά τα train-data σαν σημεία στον χώρο έτσι ώστε τα σημεία από κάθε κατηγορία να απέχουν όσο περισσότερο γίνεται, με αλλαγή του χώρου αναπαράστασης δεδομένων (πυρήνας). Τα test-data στη συνέχεια, τοποθετούνται πάνω στον ίδιο χώρο και ανάλογα σε ποια πλευρά του decision boundary πέφτουν γίνεται το classification τους σε κάποια κατηγορία.

# In[83]:


#Step 19(a)

#ΚΝΝ
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)

#Google embeddings
neigh.fit(train_google, y_train)
print(f'KNN accuracy on train_google embeddings is: {neigh.score(test_google,y_test)}')

#TF-IDF embeddings
neigh.fit(train_tfidf, y_train)
print(f'KNN accuracy on train_tfidf embeddings is: {neigh.score(test_tfidf,y_test)}')


# In[84]:


#Step 19(a)

#SVM
from sklearn import svm
clf = svm.SVC(gamma='auto')

#Google embeddings
clf.fit(train_google, y_train)
print(f'SVM accuracy on train_google embeddings is: {clf.score(test_google,y_test)}')

#TF-IDF embeddings
clf.fit(train_tfidf, y_train)
print(f'SVM accuracy on train_tfidf embeddings is: {clf.score(test_tfidf,y_test)}')


# ***Συμπεράσματα***
# 
# Παρατηρούμε λοιπόν ότι για τον KNN Classifier έχουμε καλύτερο accuracy για τα μη σταθμισμένα embeddings ενώ για τον SVM έχουμε καλύτερο accuracy για τα σταθμισμένα με TF-IDF embeddings.

# (β)
# 
# Το FastText είναι μια επέκταση του Word2Vec που προτάθηκε από το Facebook το 2016. Αντί να δίνονται ως είσοδος στο νευρωνικό ολόκληρες οι λέξεις, τις "σπάνε" σε κάποια n-grams (υπο-λέξεις). Για παράδειγμα η λέξη apple σπάει σε 3-gram ως εξής: app, ppl, και ple. Το embedding αυτής της λέξης θα είναι το άθροισμα των 3-grams (ή n-grams γενικότερα) για αυτή. Μόλις εκπαιδεύσουμε το νευρωνικό θα έχουμε word embeddings για κάθε ένα από τα n-grams στο δεδομένο dataset. Οι σπάνιες λέξεις θα μπορούν πλέον να αναπαρασταθούν καλύτερα αφού είναι αρκετά πιθανό κάποιο από τα n-grams τους να εμφανιστεί σε κάποια άλλη λέξη. 
# 
# Για αυτό το βήμα θα χρησιμοποιήσουμε ήδη εκπαιδευμένα embeddings τα οποία κατεβάσαμε από την σελίδα https://fasttext.cc/docs/en/english-vectors.html .
# 
# Δυστυχώς δεν καταφέραμε να φορτώσουμε τα embeddings λόγω προβλημάτων μνήμης , και έτσι αφήνουμε χωρις output το παρακάτω κελι. Μια εναλλακτή προσέγγιση θα ήταν να εκπαιδεύσουμε τα δικά μας fast text πάνω στο book corpus.

# In[ ]:


#Step 19(b)

from gensim.models.fasttext import  FastTextKeyedVectors

import io

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore' , limit = NUM_W2V_TO_LOAD )
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

model_fast = load_vectors("./data/wiki-news-300d-1M.vec")

