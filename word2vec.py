import numpy as np 
import pickle
import csv 
def preprocessing():
    # Stripping
    strp = ["is","a","the","this","that","of","to","an","are",'']
    symbols = ['?','!',',','"',"-","--","_",".",":",";",'']
    with open("simpsons_dataset.csv",'r+',encoding="utf8") as f:
        dataset = [" ".join([ "".join(list(map(lambda str1 : str1 if str1 not in symbols else '',list(x)))) for x in l[1].split() if x not in strp ]) for l in csv.reader(f)]
    # tokenize 
    corpus = [words.split() for words in dataset]
    print(corpus[:10])
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
        words_bag= [[i,words] for words in enumerate(self.words_bag) if words not in context]
        freq = {}
        for word in words_bag:
            freq[word[1]] = freq.setdefault(word[1],0)+1
        neg_samples = np.array(freq.values())**(3/4)/np.sum(np.array(freq.values())**(3/4))
        samples = []
        for n,val in enumerate(freq.keys()):
            samples.extend([[i[0],neg_samples[n]]for i in words_bag if i[1]==val])

        # you have to sort these neg_samples in accordance to their corresponding index so as to ultimately get a list words 
        sorted_index = sorted(samples,key=lambda x:x[1],reverse=True)
        Cindx = [index[0] for index in sorted_index][:4]
        # create the one hot encoding of the words so obtained 
        Cneg = np.zeros(len(self.words_bag))
        for i in Cindx: 
            Cneg[i]=1
        return Cneg

    def backpropagation(self,output_pos,output_neg,embeddings,weights_vector):
        sigmoid = lambda v : 1/(1+np.exp(-v))
        #loss = -np.log(output_pos)-np.sum(np.log(-sigmoid(output_neg)),axis=1)
        del_weightsPos = (sigmoid(np.dot(embeddings,output_pos.T))-1)*embeddings
        prod_neg = np.dot(embeddings,output_neg)
        embeddings_reshape = np.tile(embeddings,(len(prod_neg,len(embeddings))))
        del_weightsNeg = sigmoid(prod_neg)*embeddings_reshape
        del_embeddings = (sigmoid(np.dot(embeddings,output_pos))-1)*output_pos + np.dot(prod_neg,output_neg.T)
        self.embeddings[self.index_dict["embedding"]]-=del_embeddings
        self.weights_vector[self.index_dict["outputpos"]]-=del_weightsPos
        self.weights_vector[:,self.index_dict["outputneg"]]-=del_weightsNeg

    def run(self):
        for i in range(self.uniq_words):




            
































