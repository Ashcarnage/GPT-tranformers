import numpy as np 
import dill

''' DECORATORS EXAMPLE '''
# def decorate(func):
#     def apply(*args):
#         print("top sandwich")
#         a = func(*args)
#         print("bottom sandwich")
#         return a*2
        
#     return apply
# decorate = lambda func : lambda a : (print("topsandwich"),func(a*2),print("bottomsandwich"))[1]

# #@lambda func : lambda a : (print("topsandwich"),func(a*2),print("bottomsandwich"))[1]
# @decorate
# def cheese(a):
#     print("cheese")
#     return a


# print(cheese(2))

#print(f"a = {a}")

class convolution():   # def __init__(self,inputs,features,kernal_shape):
    def __init__(self,inputs,features,kernal_shape):
        self.inputs = np.array(inputs)
        self.features = features
        self.kernal_shape = kernal_shape
        self.kernals = np.random.randn(self.features,self.inputs.shape[0],self.kernal_shape,self.kernal_shape)
        self.bias = np.random.randn(self.features,self.inputs.shape[1] - self.kernal_shape+1,self.inputs.shape[1] - self.kernal_shape+1)
        self.filterate_shape = self.inputs.shape[1] - self.kernal_shape+1
        self.max_mat = []
        self.sigmoid = lambda x : 1/(1+np.exp(-x))
        self.sig_dev = lambda x : x*(1-x)

    @staticmethod
    def max_padding(func):
        def wrapper(*args):
            pooled_vals = func(*args)
            add_pad = lambda x,n : np.concatinate(x,np.zeros(x.shape[0]),axis = n)
            output = []
            if pooled_vals.shape[1]%2!=0:
                for vals in pooled_vals:
                    vals = add_pad(vals,0)
                    vals = add_pad(vals,1)
                    output.append(vals)
                return output
            else:
                return pooled_vals
        return wrapper
        
    def maxpool(func):
        def wrapper(self,*args):
            maxscout = lambda a,l : (l.append(np.where(a==a.max())),a.max())[-1]
            max_pooling = lambda x,l : (((maxscout(x[row:row+2,col:col+2],l)for col in range(0,len(x),2))for row in range(0,len(x),2)),self.max_mat.append(l))[1]
            conv_output = func(self,*args)
            return np.array(list(map(lambda x : max_pooling(np.array(x),l=[]),conv_output)))
        return wrapper
    
    @maxpool
    @max_padding
    def Convolve(self,inputs):
        cross_corr = lambda x,y : np.array(list(map(lambda y1 : ((np.sum(x[row:row+self.kernal_shape,col:col+self.kernal_shape]*y1)for col in range(self.filterate_shape))for row in range(self.filterate_shape)),y)))
        conv_outputs = [np.array(np.sum(np.array(list(map(lambda channel : cross_corr(np.array(channel),kernal),inputs))),axis = 1)) + self.bias[i] for i,kernal in enumerate(self.kernals)]
        return self.sigmoid(np.array(conv_outputs))
    
    
    def forward(self):
        return np.array(self.Convolve(self.inputs))


    def backward(self,output_grads,LR):
        def maxPool_invert(output_grad):
            def assign(x):
                mat = np.zeros((2,2))
                mat[x[1][0]][x[1][1]] = x[0]
                return mat
            retrace = lambda z : np.concatinate(*map(lambda x : assign(x),z),axis = 1)
            zipped3d = map(lambda output : (zip(row,self.max_mat[output[0]][i*len(row):(i+1)*len(row)]) for i,row in enumerate(output[1])),output_grad)
            outputs = map(lambda zip2d : np.concatinate(*(retrace(z) for z in zip2d),axis = 0),zipped3d)
            return np.array(list(outputs))
        
        def weight_cross(output_grad):
            corr = lambda x,y : np.sum(x*y)
            weights_grad = (map(lambda input1 : ((corr(np.array(input1)[row:row+len(grads),col:col + len(grads)],np.array(grads))for col in range(len(input1)-len(grads+1)))for row in range(len(input1)-len(grads+1))),self.inputs) for grads in output_grad)
            return np.array(list(weights_grad))
        
        def full_cross(self,output_grads):
            conv = lambda x,y : np.sum(x,y)
            def fullpad(output_grad):
                row_x = lambda x : np.concatinate(np.zeros(len(x)),x,np.zeros(len(x)),axis = 1)
                addpadding = lambda x : np.concatinate(np.zeros(len(row_x(x))),row_x(x),np.zeros(len(row_x(x))),axis = 0)
                for _ in range(self.kernel_shape):
                    output_grad = np.array(addpadding(output_grad))
                return np.array(output_grad)
            input_grads = list(map(lambda kernal : ((np.array(conv(fullpad(output))[row:row+len(kernal),col:col+len(kernal)],np.rot90(np.rot90(kernal)))for col in range(len(output)-len(kernal)+1))for row in range(len(output)-len(kernal)+1)) ) for output in output_grads)
            return np.array(input_grads)
        
        output_grads = maxPool_invert(self.sig_dev(output_grads)*output_grads)
        delta_kernals = weight_cross(output_grads)
        delta_bias  = np.copy(np.array(output_grads))
        delta_inputs = full_cross(output_grads)

        self.kernals-=LR*delta_kernals
        self.bias-=LR*delta_bias

        return delta_inputs
    
class Conv2D():
    layer_instance = []

    def __init__(self,inputs,features,kernal_shape):
        self.inputs = inputs.reshape(inputs.shape[0],1,inputs.shape[1],inputs.shape[2])
        self.kernal_count = features
        self.kernal_shape = kernal_shape
        self.layers = 1
        self.output = self.Constructframe(self)

    @classmethod
    def Constructframe(cls,self):
        relay = lambda x,l : (cls.layer_instance.append(l),x)[-1]
        def const (inputs,layer_instance):
            instance = convolution(inputs,self.kernal_count,self.kernal_shape)
            outputs = instance.forward()
            layer_instance.append(instance)
            while outputs.shape[1]<8:
                instance = convolution(outputs,self.kernal_count,self.kernal_shape)
                outputs = instance.forward()
                layer_instance.append(instance)
                self.layer+=1
            return outputs,layer_instance
        outputs = np.array(list(map(lambda inputs : relay(*const(inputs,layer_instance=[])),self.inputs)))
        outputs = outputs.reshape(outputs.shape[0],outputs.shape[1]**2)
        return outputs
    
    @classmethod
    def feedforward(cls,self):
        def forward (inputs,layer_instance):
            for instance in layer_instance:
                instance.inputs = inputs
                inputs = instance.foward()
            return inputs
        outputs = np.array(list(map(lambda inputs : forward(np.array(inputs[1]),cls.layer_instance[inputs[0]]),enumerate(self.inputs))))
        outputs = outputs.reshape(outputs.shape[0],outputs.shape[1]**2)
        return outputs
    @classmethod
    def backprop(cls,self,output_grads,LR):
        def backward(output_grads,layer_instance):
            for instance in layer_instance:
                output_grads = instance.backwards(output_grads,LR)
        map(lambda output_grad : backward(output_grads[1],cls.layer_instance[len(output_grads)-output_grad[0]-1]),enumerate(output_grads))

class ActivationFunctions:
    def __init__(self):
        self.activations = {"Sigmoid":self.Sigmoid,"Leaky_relu":self.leaky_relu,"Parametric_relu":self.parametric_relu,"Softmax":self.softmax}
    @classmethod
    def Sigmoid(cls,func):
        def wrapper(*args):
            outputs = func(*args)
            return 1/(1+np.exp(np.array(outputs))),np.array((outputs*(1-outputs)))
        return wrapper
    
    @classmethod
    def leaky_relu(cls,func):
        def wrapper(*args):
            outputs = func(*args)
            return np.where(outputs>=0,lambda x : x,lambda x : 0.01*x),np.where(outputs>=0, 1, 0.01)
        return wrapper
    
    @classmethod
    def parametric_relu(cls,alpha):
        def parametric_relu_decorator(func):
            def wrapper(*args):
                outputs = func(*args)
                return np.where(outputs>=0,lambda x : x,lambda x : alpha*x ),np.where(outputs>=0,1,alpha)
            return wrapper
        return parametric_relu_decorator
    
    @classmethod
    def softmax(cls,func):
        def wrapper(*args):
            outputs = func(*args)
            def checker(x):
                if (x[0]+1)/(x[1]+1)==1:
                    return outputs[*x]*(1-outputs[*x])
                else:
                    return np.array(outputs[*x])*np.array(outputs[x[0],x[0]])
            mat = np.where(outputs)
            return np.exp(outputs)/np.sum(np.exp(outputs)),np.array([checker(output) for output in mat ])
        return wrapper 
    
class Dense:
    Act = ActivationFunctions()
    ActFun = Act.activations
    activation = None
    def __init__(self,inputs,neurons,activation):
        self.inputs = np.array(inputs)
        self.weights = np.random.rand(self.inputs.shape[1],neurons)  
        Dense.activation = activation
        print(Dense.activation)
        self.output_devs = None
        
    #@ActFun[activation]
    def forward(self):
        self.outputs = np.dot(self.inputs,self.weights)
        return self.outputs
    
    def backward(self,output_grads,lr):
        weights_grad = np.dot(np.array((self.output_devs*output_grads)).T,self.inputs)/len(self.inputs)
        inputs_grad = np.dot(self.output_devs*output_grads,self.weights[1:,:].T)/len(self.inputs)
        self.weights-=lr*weights_grad
        return np.array(inputs_grad)

class ANN:
    layer_instance = []
    def __init__(self,inputs,neurons,layers):
        self.inputs = inputs
        self.neurons = neurons
        self.layers = layers
        self.feedforward()

    def feedforward(self):
        outputs = self.inputs
        for _ in range(self.layers-1):
            synapse = Dense(outputs,self.neurons,"Leaky_relu")
            outputs,output_devs = synapse.forward()
            synapse.output_devs = output_devs
            ANN.layer_instance.append(synapse)
        synapse = Dense(outputs,len(self.layers[1]),"Softmax")
        synapse.forward()
        ANN.layer_instance.append(synapse)
    
    def forwardprogate(self):
        for synapse in ANN.layer_instance:
            outputs,output_devs = synapse.forward()
            synapse.inputs = outputs
            synapse.output_devs = output_devs
        return np.array(outputs)
    
    def backpropgate(self,predicted,labels,lr):
        output_grad = lambda x,y : (x-y)/(x*(1-x)*len(labels)) 
        output_grad = output_grad(predicted,labels)
        for synapse in ANN.layer_instance[::-1]:
            output_grad = synapse.backward(output_grad,lr)
        return output_grad

class CNN:
    def __init__(self,data,labels):
        self.inputs = np.array(data.reshape(data.shape[0],1,data.shape[1],data.shape[2]))
        self.labels = labels
    def fit(self,lr):
        Cost = lambda x,y : -(y*np.log(x)+(1-y)*np.log(1-x))/len(self.labels)
        Convolve = Conv2D(self.inputs[0],features = 10,kernal_shape=3)
        Connect = ANN(inputs = Convolve.output,neurons = 100,layers = 5)
        loss = None
        for i in range(len(self.inputs)):
            while loss>0.01:
                outputs = Convolve.feedforward(Convolve)
                Connect.inputs = outputs
                final_outputs = Connect.forwardprogate()
                output_grad = Connect.backpropgate(final_outputs,self.labels,lr)
                Convolve.backprop(output_grad,lr)
                loss = Cost(final_outputs,self.labels)
                print(f"Model loss  : {loss}")
            Convolve.inputs = self.inputs[i+1]

import pickle 
Mnist  =  pickle.load(open("C:\\Users\\bhaka\\OneDrive\\Desktop\\MANAS\\Neural_Networks\\dataset.pkl",'rb'))
data = np.array(Mnist['data'])/255

depth = 1
l = np.array(Mnist['labels'])

a = 0;b=0
index = []
labels= np.zeros([60000,10])

newlabels = []
index = []
a,b,c= 0,0,0
failed_index = []
for label in l:
    if a<10:
        if label==a:
            bgg = list(np.take(l,failed_index))
            if label in bgg:
                index.append(failed_index[bgg.index(label)])
                failed_index.pop(bgg.index(label))
                failed_index.append(b)
                labels[b][a] = 1
            else:
                index.append(c)
                labels[b][a]=1
            a+=1
        else:
            failed_index.append(b)
            b-=1
    else:
        a=0
        b-=1
    b+=1
    c+=1

new_data1=np.take(data[:,],index,axis = 0)
new_data = new_data1[:30]
test_data = new_data1[60:]


test = CNN(new_data,labels[:10])
test.fit(lr = 0.25)


        

         



                


    
    






