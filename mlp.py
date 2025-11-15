import numpy as np

#load the arrays
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_val = np.load('X_val.npy')
y_val = np.load('y_val.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

print(X_train.shape)
print(y_train.shape)

def relu(x):
    return np.maximum(0,x)

def relu_grad(x):
    return (x > 0).astype(float)

def leaky(x, gamma):
    return np.maximum(0,x) + gamma * np.minimum(0,x)

def leaky_grad(x, gamma):
    if x > 0:
        return 1
    return gamma

def tanh_grad(x):
    return 1 - np.tanh(x) ** 2

def softmax(x):
    #subtract max for numerical stability
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)


class Node:
    def __init__(self, h, w, prev, next_layer, layer):
        self.h = h #activation function
        self.w = w #weight
        self.prev_layer = prev
        self.next_layer = next_layer
        self.layer = layer

    def activate(self,x):
        return self.h(self.w * x)
        
#10 output classes
#784 input nodes
#use He initialization for ReLU and leaky ReLU and use Xavier for tanh and softmax
#use softmax for output classification
class MLP:
    def __init__(self, h, depth, m):
        self.h = h
        self.depth = depth
        self.m = m
        self.layers = []

        # Determine initialization based on activation
        def get_init_scale(n_in, n_out):
            if h == relu or h == leaky_relu:
                return np.sqrt(2 / n_in)
            else:  # tanh
                return np.sqrt(1 / (n_in + n_out))

        #input layer
        n_in = 784
        n_out = m
        scale = get_init_scale(n_in, n_out)
        weights = np.random.normal(0, scale, size=(n_in, m))

        current_layer = []
        for i in range(n_in):
            current_layer.append(Node(h, weights[i], None, None, 0))
        self.layers.append(current_layer)

        #hidden layers
        for j in range(depth):
            current_layer = []
            n_in = m
            scale = get_init_scale(n_in, n_out)
            weights = np.random.normal(0, scale, size=(n_in, m))

            for i in range(m):
                current_layer.append(Node(h, weights[i], self.layers[j], None, j + 1))

            self.layers.append(current_layer)

        #output layer (always softmax, use Xavier)
        n_in = m
        n_out = 10
        scale = np.sqrt(1 / (n_in + n_out))  # Xavier for softmax
        weights = np.random.normal(0, scale, size=(n_in, 10))

        current_layer = []
        for i in range(10):
            current_layer.append(Node(softmax, weights[i], self.layers[depth], None, depth + 1))

        self.layers.append(current_layer)

        #connect layers
        for layer_idx in range(len(self.layers) - 1):
            for node in self.layers[layer_idx]:
                node.next_layer = self.layers[layer_idx + 1]

    def fit(self, x):
        #TODO
        return
    def prdict(self, x):
        #TODO
        return

model = MLP(relu,2,50)
