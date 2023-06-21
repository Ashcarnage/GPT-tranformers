
import numpy as np 
#   ///////////// PREREQUISUTS ///////////// 
class layers:
    def __init__(self,inputs,neurons,activation = "relu"):
        self.inputs = np.append(1,np.array(inputs))
        self.neurons = neurons
        self.weights = np.random.rand(len(self.inputs),self.neurons)
        self.m = np.zeros([len(self.inputs),self.neurons])
        self.v = np.copy(self.m)

    def forward(self):
        relu = lambda x : np.where(x>=0,x,0.01*x)
        self.outputs = np.array(relu(np.dot(self.inputs,self.weights)))
        return self.outputs
    def backward(self,output_gradident):
        relu_dev = lambda x : np.where(x>=0,1,0.01)#1 if x>=0 else 0.01
        self.delta_weights = np.dot(self.inputs.T,relu_dev(self.outputs))*output_gradident
        delta_input = np.dot(self.weights[1:,:],relu_dev(self.outputs))*output_gradident
        return np.array(delta_input)


class Dense:
    layer_instance = []
    weights_repo = []
    def __init__(self,inputs,neurons=100,layers=4,activation="relu"):
        self.inputs = inputs
        self.neurons = neurons
        self.layers = layers
        self.activation = activation
    def build(self):
        output = self.inputs
        for layer in self.layers:
            if layer!=self.layers-1:
                synapse = layers(output,self.neurons,self.activation)   
            else:
                synapse = layers(output,len(output),self.activation)
            output = synapse.forward() 
            Dense.layer_instance.append(synapse)
        return output
    def feedforward(self):
        output = self.inputs
        for layer in Dense.layer_instance:
            layer.inputs = output
            output = layer.forward()
    def backprop(self,x,LR):
        softmax = lambda x : np.exp(x)/np.sum(np.exp(x))
        grid = np.indicies(x.shape)
        output_gradient  = softmax(x)*(np.where(grid[0]==grid[1],1,0))
        for layer in Dense.layer_instance[::-1]:
            output_gradient = layer.backward(output_gradient)
            del_weights = np.array(layer.delta_weights)
            layer.m = (0.9*layer.m + 0.1*del_weights)/0.1
            layer.v = (0.999*layer.v+0.001*del_weights)/0.001
            layer.weights-=LR*layer.m/(np.sqrt(layer.v)+0.1/10**6)
            Dense.weights_repo.append(layer.weights)



            

            


                
