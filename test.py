# numul = lambda x,y : (x*y,x/y)
# def fadone():
#     a= 3 
#     b = 4
#     return a, b
# import numpy as np 
# # print([*fadone()])
# # print(numul(*fadone()))
# l=np.array([[9,8,7,6,5,4],[9,8,7,6,5,4]])
# # output = lambda x : x[1]
# # for i in map(lambda x : output(x),enumerate(l[::-1])):
# #     print(i)
# mat = np.where(l)
# #for i in zip(np.array(mat).T) : print(l[*i[0]])
# print(np.array(mat).T)
# def deco1(func):
#     def wrapper(*args):
#         a = func(*args)
#         print("made your way right at the very deco1 you created ")
#     return wrapper

# def deco2(func):
#     def wrapper(*args):
#         a = func(*args)
#         print("quite the test wasnt it ... seems like you failed ")
#     return wrapper
# l = [deco1,deco2]

# def mains(a):
#     for i in range(a):
#         print("lets get to it ,whatever happens we'll deal with it as we always do")
# mains(1)



'''
import numpy as np 
import dill

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
        
        
    def maxpool(self,func):
        def wrapper(*args):
            maxscout = lambda a,l : (l.append(np.where(a==a.max())),a.max())[-1]
            max_pooling = lambda x,l : (((maxscout(x[row:row+2,col:col+2],l)for col in range(0,len(x),2))for row in range(0,len(x),2)),self.max_mat.append(l))[1]
            conv_output = func(*args)
            return map(lambda x : max_pooling(np.array(x),l=[]),conv_output)
        return wrapper
    
    @maxpool
    @max_padding
    def Convolve(self,inputs):
        cross_corr = lambda x,y : map(lambda y1 : ((np.sum(x[row:row+self.kernal_shape,col:col+self.kernal_shape]*y1)for col in range(self.filterate_shape))for row in range(self.filterate_shape)),y)
        conv_outputs = (np.sum(map(lambda channel : cross_corr(np.array(channel),kernal),inputs),axis = 1) + self.bias[i] for i,kernal in enumerate(self.kernals))
        return self.sigmoid(np.array(conv_outputs))
    
    
    def forward(self,inputs):
        return np.array(self.Convolve(inputs))


    def backward(self,output_grads,LR):
        def maxPool_invert(output_grad):
            def assign(x):
                mat = np.zeros((2,2))
                mat[x[1][0]][x[1][1]] = x[0]
                return mat
            retrace = lambda z : np.concatinate(*map(lambda x : assign(x),z),axis = 1)
            zipped3d = map(lambda output : (zip(row,self.max_mat[output[0]][i*len(row):(i+1)*len(row)]) for i,row in enumerate(output[1])),output_grad)
            outputs = map(lambda zip2d : np.concatinate(*(retrace(z) for z in zip2d),axis = 0),zipped3d)
            return np.array(outputs)
        
        def weight_cross(output_grad):
            corr = lambda x,y : np.sum(x*y)
            weights_grad = (map(lambda input1 : ((corr(np.array(input1)[row:row+len(grads),col:col + len(grads)],np.array(grads))for col in range(len(input1)-len(grads+1)))for row in range(len(input1)-len(grads+1))),self.inputs) for grads in output_grad)
            return weights_grad
        
        def full_cross(self,output_grads):
            conv = lambda x,y : np.sum(x,y)
            def fullpad(output_grad):
                row_x = lambda x : np.concatinate(np.zeros(len(x)),x,np.zeros(len(x)),axis = 1)
                addpadding = lambda x : np.concatinate(np.zeros(len(row_x(x))),row_x(x),np.zeros(len(row_x(x))),axis = 0)
                for _ in range(self.kernel_shape):
                    output_grad = np.array(addpadding(output_grad))
                return np.array(output_grad)
            input_grads = (map(lambda kernal : ((np.array(conv(fullpad(output))[row:row+len(kernal),col:col+len(kernal)],kernal)for col in range(len(output)-len(kernal)+1))for row in range(len(output)-len(kernal)+1)) ) for output in output_grads)
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
        self.Constructframe(self)

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
        outputs = map(lambda inputs : relay(*const(inputs,layer_instance=[])),self.inputs)
        return np.array(outputs)
    
    @classmethod
    def feedforward(cls,self):
        def forward (inputs,layer_instance):
            for instance in layer_instance:
                instance.inputs = inputs
                inputs = instance.foward()
            return inputs
        outputs = map(lambda inputs : forward(np.array(inputs[1]),cls.layer_instance[inputs[0]]),enumerate(self.inputs))
        return outputs
    @classmethod
    def backprop(cls,self,output_grads,LR):
        def backward(output_grads,layer_instance):
            for instance in layer_instance:
                output_grads = instance.backwards(output_grads,LR)
        map(lambda output_grad : backward(output_grads[1],cls.layer_instance[len(output_grads)-output_grad[0]-1]),enumerate(output_grads))

class ActivationFUnctions:
    state = 1
    def __init__(self):
        self.activations = [self.Sigmoid,self.leaky_relu,self.parametric_relu,self.softmax]
    @classmethod
    def Sigmoid(cls,func):
        def wrapper(*args):
            outputs = func(*args)
            if cls.state:
                    return 1/(1+np.exp(np.array(outputs)))
            else: 
                    return np.array(outputs*(outputs*(1-outputs)))
        return wrapper
    @classmethod
    def leaky_relu(cls,func):
        def wrapper(*args):
            outputs = func(*args)
            if cls.state:
                return np.where(outputs>=0,lambda x : x,lambda x : 0.01*x)
            else : 
                return np.where(outputs>=0, 1, 0.01)
        return wrapper
    @classmethod
    def parametric_relu(cls,alpha):
        def parametric_relu_decorator(func):
            def wrapper(*args):
                outputs = func(*args)
                if cls.state:
                    return np.where(outputs>=0,lambda x : x,lambda x : alpha*x )
                else:
                    return np.where(outputs>=0,1,alpha)
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
            if cls.state :
                cls.state = 0
                return np.exp(outputs)/np.sum(np.exp(outputs))
            else:
                mat = np.where(outputs)
                return np.array([checker(output) for output in mat ])
        return wrapper 
class Dense:
    def __init__(self,inputs,neurons,labels=None):
        self.inputs = np.array(inputs)
        self.weights = np.random.rand(self.inputs.shape[1],neurons)    

    def feedforard(self):
        self.outputs = np.dot(self.inputs,self.weights)
        return self.outputs
    
    def backprop(self,output_grads,lr):
        weights_grad = np.dot(np.array(output_grads).T,self.inputs)/len(self.inputs)
        inputs_grad = np.dot(output_grads,self.weights[1:,:].T)/len(self.inputs)
        self.weights-=lr*weights_grad
        return np.array(inputs_grad)
        
'''
# import numpy as np 
# def test():
#     get_val = lambda x: np.where(x>=0,x,0.01*x)
#     print(get_val(np.array([0,-1,1,2,-9])))
# test()

import csv 
strp = ["is","a","the","this","that","of","to","an","are",'']
with open("simpsons_dataset.csv",'r+') as f:
    dataset = [" ".join([ "".join(filter(lambda str : str not in ['?','!',',','"',"-","--","_",".",":",";",''],x))for x in l[1].split() if x not in strp ]) for l in csv.reader(f)]
print(dataset[:200])

