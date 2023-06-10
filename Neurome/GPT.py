
import numpy as np 

class layers:
    def __init__(self,inputs,neurons,activation = "relu"):
        self.inputs = np.append(1,np.array(inputs))
        self.weights = np.random.rand(len(self.inputs),neurons)
    def forward(self,last = False):
        relu = lambda x : np.where(x>=0,x,0.01*x)
        softmax = lambda x : np.exp(x)/np.sum(np.exp(x))
        self.outputs = np.array(softmax(np.dot(self.inputs,self.weights))) if last else np.array(relu(np.dot(self.inputs,self.weights)))
        return self.outputs
    def backward(self,output_gradident):
        relu_dev = lambda x : np.where(x>=0,1,0.01)
        self.delta_weights = np.dot(self.inputs.T,relu_dev(self.outputs))*output_gradident
        delta_input = np.dot(self.weights[1:,:],relu_dev(self.outputs))*output_gradident
        return np.array(delta_input)


class Dense:
    layer_instance = []
    def __init__(self,inputs,neurons=100,layers=4,activation="relu"):
        self.inputs = inputs
        self.neurons = neurons
        self.layers = layers
        self.activation = activation
    def build(self,last = False):
        output = self.inputs
        for layer in self.layers:
            if layer!=self.layers-1:
                synapse = layers(output,self.neurons,self.activation)  
            else:
                synapse = layers(output,len(output),self.activation)
                last = True
            output = synapse.forward(last) 
            Dense.layer_instance.append(synapse)
        return output
    def feedforward(self):
        output = self.inputs
        for layer in Dense.layer_instance:
            layer.inputs = output
            output = layer.forward()
    def backprop(self):
        #softmax derivative 

        output_gradient  = 
        for layer in Dense.layer_instance[::-1]:
            output_gradident = layer.backward(output_gradident)


                
