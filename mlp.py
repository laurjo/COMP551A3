import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Load the arrays
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
#the validation set could be used for cross-validation if we decide to implement it
#X_val = np.load('X_val.npy')
#y_val = np.load('y_val.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

print("Dataset shapes:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
#print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
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

    def fit(self, X_train, y_train, X_test, y_test, learning_rate=0.01, epochs=100, batch_size=64):
        """
        Train the MLP using mini-batch gradient descent
        """
        n_samples = X_train.shape[0]
        n_batches = n_samples // batch_size

        train_losses = []
        train_accuracies = []
        test_accuracies = []

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

            yh = self.predict(X_train)
            train_acc = self.evaluate_acc(y_train,yh)
            train_accuracies.append(train_acc)

            #test/validation
            yh = self.predict(X_test)
            test_acc = self.evaluate_acc(y_test,yh)
            test_accuracies.append(test_acc)

        return train_losses, train_accuracies, test_accuracies

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

def cross_validation_split(n, n_folds=5):
    #get the number of data samples in each split
    n_val = n // n_folds
    inds = np.random.permutation(n)
    inds = []
    for f in range(n_folds):
        tr_inds = []
        #get the validation indexes
        val_inds = list(range(f * n_val, (f+1)*n_val))
        #get the train indexes
        if f > 0:
            tr_inds = list(range(f*n_val))
        if f < n_folds - 1:
            tr_inds = tr_inds + list(range((f+1)*n_val, n))
        #The yield statement suspends function’s execution and sends a value back to the caller
        #but retains enough state information to enable function to resume where it is left off
        yield tr_inds, val_inds

def kfold_cross_val(x , y, n_folds , model, lr, n_epoches):
    score_val = np.zeros(n_folds)
    for f, (tr, val) in enumerate(cross_validation_split(x.shape[0], n_folds)):
        train_losses, train_accuracies, test_accuracies = model.fit(x[tr], y[tr], x[val], y[val], learning_rate=lr, epochs=n_epoches)
        score_val[f] = model.evaluate_acc(y[val], model.predict(x[val]))
    return score_val, score_val.mean()


#initial tuning for #of epochs:
# for i in range(3):
#     model = MLP(relu, depth=i, m=256)
#     train_losses, train_accuracies, test_accuracies = model.fit(X_train, y_train, X_test, y_test,learning_rate=0.01,epochs=100)
#     plt.plot(train_losses)
#     plt.ylabel("Training Loss")
#     plt.xlabel("Epochs")
#     plt.title("Hidden Layers = "+ str(i))
#     fig, ax = plt.subplots(figsize=(5, 4), layout='constrained')
#     ax.plot(test_accuracies, label="Testing Set")
#     ax.plot(train_accuracies, label = "Training Set")
#     ax.set_xlabel("Epochs")
#     ax.set_ylabel("Accuracy")
#     ax.set_title("Hidden Layers = "+ str(i))
#     ax.legend()
#     plt.show()

#k-fold cross validation for learning rate:
#learing_rate = np.array([1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1])

# loop over learning rates and number of hidden layers
# for x in range(3):
#     mn = MLP(relu, depth=2, m=256)
#     best_score = 0
#     n=0
#     accs=np.zeros(len(learing_rate))
#     for i in learing_rate:
#             score = kfold_cross_val(X_train,y_train, 5, mn, i, 60)[1]
#             accs[n]= score
#             n+=1
#             if score > best_score:
#                 best_score = score
#                 best_lr = i
#             print(f'for learning_rate= {i}  => score = {score}, best score = {best_score}')
#     print(f'Best learning_rate: {best_lr}')

#     plt.plot(learing_rate, accs)
#     plt.ylabel('Accuracy')
#     plt.xlabel('Learning Rate')
#     plt.show()

opt_rate = [0.005, 0.01, 0.01]
for i in range(3):
    model = MLP(relu, depth=i, m=256)
    train_losses, train_accuracies, test_accuracies = model.fit(X_train, y_train, X_test, y_test,learning_rate=opt_rate[i],epochs=100)
    plt.plot(train_losses)
    plt.ylabel("Training Loss")
    plt.xlabel("Epochs")
    plt.title("Hidden Layers = "+ str(i))
    fig, ax = plt.subplots(figsize=(5, 4), layout='constrained')
    ax.plot(test_accuracies, label="Testing Set")
    ax.plot(train_accuracies, label = "Training Set")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.set_title("Hidden Layers = "+ str(i))
    ax.legend()
    plt.show()
    yh_train = model.predict(X_train)
    yh_test = model.predict(X_test)
    test_acc = model.evaluate_acc(y_test, yh_test)
    train_acc = model.evaluate_acc(y_train, yh_train)
    print(f"{i} Hidden Layers:")
    print(f"Final test acc = {test_acc}")
    print(f"Final test acc = {train_acc}")