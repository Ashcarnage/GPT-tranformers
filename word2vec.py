import numpy as np 
import nltk 
# nltk.download('punkt')
import pickle
import csv 

def preprocessing():
    # Stripping
    strp = ["is","a","the","this","that","of","to","an","are",'']
    symbols = ['?','!',',','"',"-","--","_",".",":",";",'']
    with open("simpsons_dataset.csv",'r+',encoding="utf8") as f:
        dataset = [" ".join([ "".join(list(map(lambda str1 : str1 if str1 not in symbols else '',list(x)))) for x in l[1].split() if x not in strp ]) for l in csv.reader(f)]
    # tokenize 
    corpus = [nltk.word_tokenize(words) for words in dataset]
    with open("wordcorpus.dat",'wb+') as f :
        pickle.dump(corpus,f)


data  = pickle.load(open("wordcorpus.dat",'rb'))

class skip_gram:
    def __init__(self,data):
        self.data = data
        self.words_bag = np.concatenate(self.data)
        self.uniq_words = []
        list(map(lambda word : self.uniq_words.append(word) if word not in self.uniq_words else "" ,self.words_bag))
        self.embeddings = np.random.randn(len(self.uniq_words),300)
        self.weights_vector = np.random.randn(300,len(self.uniq_words))

    def target_contextwindow(self,index,window_size=3):
        context = []
        target = np.zeros(len(self.uniq_words))
        target[index]=1
        position = index
        contextlist = [[index-window_size+a,word] for a,word in enumerate(index-window_size,index+window_size+1) if a!=index]
        for index,word in contextlist:
            contextvec = np.zeros(self.uniq_words)
            contextvec[index]=1
            context.append(contextvec)
        self.document = [target,context,contextlist]
        return self.document 

    def negative_samples(self,document):
        context = [j for i,j in document[2]]
        self.words_bag = np.array(self.words_bag)
        words_bag= [words for words in self.words_bag if words not in context]
        freq = {}
        for word in words_bag:
            freq[word] = freq.setdefault(word,0)+1
        neg_samples = np.array(freq.values())**(3/4)/np.sum(np.array(freq.values())**(3/4))
        # you have to sort these neg_samples in accordance to their corresponding index so as to ultimately get a list words 
        sorted_index = sorted(zip(range(len(neg_samples)),neg_samples),key=lambda x:x[1],reverse=True)
        Cindx = [index[1] for index in sorted_index][:4]
        # create the one hot encoding of the words so obtained 
        Cneg = np.zeros(len(self.words_bag))
        for i in Cindx: 
            Cneg[i]=1
        return Cneg

    def backpropagation(self,output_pos,output_neg):
        sigmoid = lambda v : 1/(1+np.exp(-v))
        loss = -np.log(output_pos)-np.sum(np.log(-sigmoid(output_neg)),axis=1)
            

































