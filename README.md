# Neural Network with Numpy

Design a neural network for recognition digit numbers MNIST classification datasets with 10 class from 0-9.

Build the neural network with the following `Make Your Own Neural Network` book and the `Introduction to Machine Learning` course in HCMUS.

The Architecture:

![image](https://github.com/minhtri6179/NNwithNumpy/blob/master/images/arch.jpeg)

We will build difference neural networks with any architectures we want. We can change the number of nodes each layers and the number of layer of networks.
# I. Packages
- numpy
- tensorflow
- matplotlib

# II. The Neural Network
## Architecture
I build the neural network fully connected `Dense` which each neuron connect to every other neuron in the next layer. It called Feedforward, the phase learn is update `Weight` between nodes. 

The default architecture is 4 layers: `1 input layer`, `2 hidden layer`, `1 output layer`. Activation function: `ReLU`, `DReLU` and `Sigmoid` function. Optimizer: `Mini Batch Gradient Descent`
- Layers: 
    - Input layer: the MNIST dataset is 28x28 dimensions. Firstly, it need to flatten to 1d array and each image digit is 28x28=784 pixels. Thus, input layer has 784 nodes each node represents a pixels.
    - Hidden layer: Is the layer extract feature for image digit. Each layer in hidden layer performs smaller feature of digit.
    - Output layer: is the layer show the probabilities 
- Activation function: Good for sigmoid, not designed for other activation function :(((

## Feedforward
The signal from input layer pass through next layer to the last layer. The input layer does not need apply activation function. In the hidden layers, the signal from previous layer use `matrix multiplication` the next layer. After each calculating, the result will apply to an activation function such as `sigmoid`, `ReLU` and so on. 
        
        Layeri = activation(XLayer(i-1))

## Backpropagation
The learning is update weight between layer nodes to find the minimum `error`. Finding the minimum difference between the training data and the actual output.

## Gradient Descent
Gradient descent is the method can find the value x for the f(x get minimum. The gradient descent starts somewhere from the graph, look around and the direction is downwards. It really good to get minimum of complex function, which hard to work with mathematically using algebra.

# III. Command to train the neural network
```bash
git clone https://github.com/minhtri1f9/NNwithNumpy
```

```bash
python NNwithNumpy/train.py --hidden_size 50 50 --batch_size 32 --lr 0.1 --epochs 10 
```
- `hidden_size` is the number of hidden layers in the network
- `batch_size` is the number of instances each bath
- `lr` is the learning rate 
- `epochs` is the number of epochs to train the network

Using Python script to train the network and predict result of digit image
```python
nn = NeuralNet(hidden_size=[50, 50], batch_size=32, lr=0.1, epochs=2, activation_function='relu')
x_train, y_train, x_val, y_val = nn.getMNIST()
A = nn.train(x_train, y_train, x_val, y_val)
```

Predict the new digit
```python
nn.predict(x_val[20])
```

![image](https://drive.google.com/uc?export=view&id=1ctxoo3Xd0977JeRqld-mnOEslnKff0LD)