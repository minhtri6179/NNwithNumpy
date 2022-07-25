import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)


class NeuralNet:

    def __init__(self, hidden_size, batch_size, lr, epochs, activation_function='sigmoid'):
        """
        Params:
                hidden_size: is the size of hidden layer 
                              [50, 50, 50]: 3 hidden layer, each layer has 50 node

                batch_size: is the size each mini batch

                epochs: the number of iterator dataset

                activation_function: [sigmoid, RELU], sigmoid for default
        """
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.errs = []
        self.val_errs = []
        self.cache_layer = []
        # RELU activation function for each ouput
        self.activation_function = activation_function

    def ac_fu(self, x):
        if self.activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation_function == 'relu':
            return np.maximum(x, 0)
        elif self.activation_function == 'drelu':
            return 1. * (x > 0)

    def getMNIST(self):
        keras_mnist = tf.keras.datasets.mnist.load_data(path="mnist.npz")
        (x_train, y_train), (x_val, y_val) = keras_mnist

        return x_train, y_train, x_val, y_val

    def prepare_dataset(self, X, y, x_val, y_val, normal=255.0, img_size=784,
                        one_set=False):
        """
            Params: 
                    normal: the number is devided to normalation avoid nan problem
                    img_size: is the size of image after falten
                    one_set: if true, it just return x value after preprocessing
            Return: 
                    X, y, x_val, y_val after preprocessing
                      reshape the image to 1d array
                      normalization to easy compute
        """
        # Falten
        X = X.reshape(-1, img_size)
        X = (np.asfarray(X) / normal * 0.99) + 0.01
        if one_set == True:
            return X
        x_val = x_val.reshape(-1, img_size)
        x_val = (np.asfarray(x_val) / normal * 0.99) + 0.01

        return X, y, x_val, y_val

    def feed_forward(self, X, W, cache=False):
        """
          Params: 
                  X: input matrix consist of instances, each row has each instance
                      each column is the feauture of it. value from 0-255
                  W: Weight matrix
          Return: return the final output if cache param is False and return output 
                  matrix when run feed forward process 
        """
        output_layer = X
        self.cache_layer.append(X)

        for layer_idx, layer in enumerate(W):
            output = self.ac_fu(output_layer.dot(layer))
            bias = np.ones(output_layer.shape[0]).reshape(-1, 1)
            if layer_idx == len(W)-1:
                output_layer = output
                self.cache_layer.append(output_layer)
                break

            output_layer = np.append(bias, output, axis=1)
            self.cache_layer.append(output_layer)
            self.output_layer = output_layer
        if cache == True:
            return self.cache_layer
        return output_layer

    def train(self, X, y, val_x, val_y):
        """
            Params: 
              - X: the images matrix using for train
              - y: labels for X training set
            Return:
              - Ws: is the Weight matrix of neural netword after traning
                    input layer to hidden layer - hidden layer to output layer
                    the signal will pass through activation function after each layer

        """
        print('=============Training Progress================')
        print('----------------Begin--------------------')
        # Preprocessing
        X, y, val_x, val_y = self.prepare_dataset(X, y, val_x, val_y)

        # init
        n_class = len(np.unique(y))
        layer_size = [X.shape[1] - 1] + self.hidden_size + [n_class]

        # Weight matrix
        Ws = [np.random.randn(layer_size[i] + 1, layer_size[i + 1]) / np.sqrt(layer_size[i] + 1)
              for i in range(len(layer_size) - 1)]

        # create one hot
        one_hot_Y = np.zeros((len(y), n_class))
        one_hot_Y[np.arange(len(y)), y] = 1

        rnd_idxs = np.arange(len(X))
        # The traning
        for epoch in range(self.epochs):
            np.random.shuffle(rnd_idxs)
            for start_idx in range(0, len(X), self.batch_size):
                # Mini batch
                mb_X = X[rnd_idxs[start_idx:start_idx+self.batch_size]]
                mb_Y = one_hot_Y[rnd_idxs[start_idx:start_idx+self.batch_size]]

                # Forward
                self.feed_forward(mb_X, Ws, True)
                As = self.cache_layer
                # Backprop
                delta = As[-1] - mb_Y
                mb_size = self.batch_size
                if start_idx + self.batch_size >= len(X):
                    mb_size = n_class % self.batch_size
                grad = (As[-2].T.dot(delta)) / mb_size
                Ws[-1] -= self.lr * grad
                for i in range(2, len(Ws) + 1):
                    delta = delta.dot(Ws[-i + 1].T[:, 1:]) * \
                        As[-i][:, 1:] * (1 - As[-i][:, 1:])
                    grad = (As[-i-1].T.dot(delta)) / mb_size
                    Ws[-i] -= self.lr * grad

            A = self.feed_forward(X, Ws)
            err = np.mean(np.argmax(A, axis=1) != y) * 100
            self.errs.append(err)
            val_A = self.feed_forward(val_x, Ws)
            val_err = np.mean(np.argmax(val_A, axis=1) != val_y) * 100
            acc = np.mean(np.argmax(val_A, axis=1) == val_y) * 100
            self.val_errs.append(val_err)
            print(
                f'Epoch {epoch+1}/{self.epochs}: train err: {err:.2f}%, val err: {val_err:.2f}% - accuency: {acc:.2f}%')
        print('----------------Done--------------------')
        self.Ws = Ws
        return Ws

    def predict(self, x, verbose=True):
        """
          Preprocessing instance and pass it into Weight matrix of neural network,
          using argmax to find the index have maximun value of softmax
          Set `verbose` = False to avoid show image of number and result of number
        """
        in_x = self.prepare_dataset(x, 1, 2, 3, one_set=True)
        result = self.feed_forward(in_x, self.Ws)
        if verbose == False:
            return np.argmax(result)
        print('=============The image of number================')
        plt.imshow(x)

        print(f'I guess the number is: {np.argmax(result)}')
        print('=============Done Process================')
        return np.argmax(result)
