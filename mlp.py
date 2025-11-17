import numpy as np

# Load the arrays
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
#the validation set could be used for cross-validation if we decide to implement it
X_val = np.load('X_val.npy')
y_val = np.load('y_val.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

print("Dataset shapes:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

#activation functions and their derivatives
def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def leaky_relu(x, gamma=0.01):
    return np.maximum(0, x) + gamma * np.minimum(0, x)

def leaky_relu_grad(x, gamma=0.01):
    grad = np.ones_like(x)
    grad[x < 0] = gamma
    return grad

def tanh(x):
    return np.tanh(x)

def tanh_grad(x):
    return 1 - np.tanh(x) ** 2

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def cross_entropy_loss(y_pred, y_true):
    """
    y_pred: (batch_size, num_classes)
    y_true: (batch_size,)
    """
    n = y_true.shape[0]
    #convert labels to one-hot encoding
    y_one_hot = np.zeros_like(y_pred)
    y_one_hot[np.arange(n), y_true] = 1
    #compute cross-entropy loss
    loss = -np.sum(y_one_hot * np.log(y_pred + 1e-8)) / n
    return loss

class MLP:
    def __init__(self, h, depth, m, gamma=0.01):
        """
        h: activation function
        depth: number of hidden layers
        m: number of units per hidden layer
        gamma: parameter for leaky_relu
        """
        self.h = h
        self.depth = depth
        self.m = m
        self.gamma = gamma
        
        #determine activation gradient function
        if h == relu:
            self.h_grad = relu_grad
        elif h == leaky_relu:
            self.h_grad = lambda x: leaky_relu_grad(x, gamma)
        elif h == tanh:
            self.h_grad = tanh_grad
        else:
            raise ValueError("Unknown activation function")
        
        #initialize weights and biases
        self.weights = []
        self.biases = []
        
        #determine initialization scale
        def get_init_scale(n_in, n_out):
            if h == relu or h == leaky_relu:
                return np.sqrt(2 / n_in)  #He initialization
            else:  #tanh
                return np.sqrt(1 / (n_in + n_out))  #Xavier initialization
        
        #input to first hidden layer
        n_in = 784
        if depth == 0:
            n_out = 10
            scale = np.sqrt(1 / (n_in + n_out))  # Xavier for output layer
            self.weights.append(np.random.normal(0, scale, size=(n_in, n_out)))
            self.biases.append(np.zeros((1, n_out)))
        else:
            #first hidden layer
            n_out = m
            scale = get_init_scale(n_in, n_out)
            self.weights.append(np.random.normal(0, scale, size=(n_in, n_out)))
            self.biases.append(np.zeros((1, n_out)))
            
            #init additional hidden layers
            for i in range(depth - 1):
                n_in = m
                n_out = m
                scale = get_init_scale(n_in, n_out)
                self.weights.append(np.random.normal(0, scale, size=(n_in, n_out)))
                self.biases.append(np.zeros((1, n_out)))
            
            #init output layer
            n_in = m
            n_out = 10
            scale = np.sqrt(1 / (n_in + n_out))  # Xavier for softmax output
            self.weights.append(np.random.normal(0, scale, size=(n_in, n_out)))
            self.biases.append(np.zeros((1, n_out)))

    def forward(self, x):
        """
        Forward pass through the network
        x: (batch_size, 784)
        Returns: predictions (batch_size, 10), cache for backprop
        """
        cache = {'activations': [x], 'pre_activations': []}
        
        a = x
        for i in range(len(self.weights) - 1):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            cache['pre_activations'].append(z)
            
            if self.h == leaky_relu:
                a = self.h(z, self.gamma)
            else:
                a = self.h(z)
            cache['activations'].append(a)
        
        #output layer with softmax
        z = np.dot(a, self.weights[-1]) + self.biases[-1]
        cache['pre_activations'].append(z)
        y_pred = softmax(z)
        cache['activations'].append(y_pred)
        
        return y_pred, cache
    
    def backward(self, y_pred, y_true, cache):
        """
        Backward pass to compute gradients
        y_pred: (batch_size, 10)
        y_true: (batch_size,)
        cache: forward pass cache
        """
        n = y_true.shape[0]
        
        #convert labels to one-hot
        y_one_hot = np.zeros_like(y_pred)
        y_one_hot[np.arange(n), y_true] = 1
        
        dz = (y_pred - y_one_hot) / n
        
        grads_w = []
        grads_b = []
        
        # Backprop through layers in reverse
        for i in range(len(self.weights) - 1, -1, -1):
            #gradient of weights and biases
            a_prev = cache['activations'][i]
            dw = np.dot(a_prev.T, dz)
            db = np.sum(dz, axis=0, keepdims=True)
            
            grads_w.insert(0, dw)
            grads_b.insert(0, db)
            
            #gradient of previous layer activation
            if i > 0:
                da = np.dot(dz, self.weights[i].T)
                #gradient through activation function
                z_prev = cache['pre_activations'][i - 1]
                dz = da * self.h_grad(z_prev)
        
        return grads_w, grads_b

    def fit(self, X_train, y_train, X_val, y_val, learning_rate=0.01, epochs=100, batch_size=64):
        """
        Train the MLP using mini-batch gradient descent
        """
        n_samples = X_train.shape[0]
        n_batches = n_samples // batch_size

        train_losses = []
        val_losses = []
        val_accuracies = []

        for epoch in range(epochs):
            #shuffle training data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0

            #mini-batch gradient descent
            for batch in range(n_batches):
                start = batch * batch_size
                end = start + batch_size

                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Forward pass
                y_pred, cache = self.forward(X_batch)

                #compute loss
                loss = cross_entropy_loss(y_pred, y_batch)
                epoch_loss += loss

                #backward pass
                grads_w, grads_b = self.backward(y_pred, y_batch, cache)

                #update weights and biases
                for i in range(len(self.weights)):
                    self.weights[i] -= learning_rate * grads_w[i]
                    self.biases[i] -= learning_rate * grads_b[i]

            #average loss over batches
            epoch_loss /= n_batches
            train_losses.append(epoch_loss)

            #validation
            val_pred, _ = self.forward(X_val)
            val_loss = cross_entropy_loss(val_pred, y_val)
            val_losses.append(val_loss)

            yh = self.predict(X_val)
            val_acc = self.evaluate_acc(y_val,yh)
            val_accuracies.append(val_acc)

        return train_losses, val_losses, val_accuracies

    def predict(self, X):
        """
        Make predictions and compute accuracy
        X: (n_samples, 784)
        y: (n_samples,) - true labels
        Returns: accuracy
        """
        yh, _ = self.forward(X)
        return yh

    def evaluate_acc(self,y,yh):
        predictions = np.argmax(yh, axis=1)
        accuracy = np.mean(predictions == y)
        return accuracy

model = MLP(relu, depth=1, m=50)

train_losses, val_losses, val_accuracies = model.fit(X_train, y_train, X_val, y_val,learning_rate=0.01,epochs=50,batch_size=64)

#test accuracy
yh = model.predict(X_test)
accuracy = model.evaluate_acc(y_test, yh)
print(f"\nFinal Test Accuracy: {accuracy:.4f} for model with {model.depth} hidden layers and {model.m} hidden units per layer")
