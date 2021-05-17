"""Neural network model."""

from typing import Sequence

import numpy as np


class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and performs classification
    over C classes. We train the network with a cross-entropy loss function and
    L2 regularization on the weight matrices.

    The network uses a nonlinearity after each fully connected layer except for
    the last. The outputs of the last fully-connected layer are passed through
    a softmax, and become the scores for each class."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:

        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)

        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: The number of classes C
            num_layers: Number of fully connected layers in the neural network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers

        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(
                sizes[i - 1], sizes[i]
            ) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])
            self.params["m" + str(i)] = np.zeros([sizes[i-1], sizes[i]])
            self.params["v" + str(i)] = np.zeros([sizes[i-1], sizes[i]])

    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.

        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias

        Returns:
            the output
        """
        # TODO: implement me
        out = X @ W + b
        return out

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).

        Parameters:
            X: the input data

        Returns:
            the output
        """
        # TODO: implement me
        
        out = X*(X>0)
        return out 

    def softmax(self, X: np.ndarray) -> np.ndarray:
        """The softmax function.

        Parameters:
            X: the input data

        Returns:
            the output
        """
        # TODO: implement me
        exp = np.exp(X - np.max(X,axis=1).reshape(X.shape[0],1))
        out = exp/(np.sum(exp,axis=1).reshape(exp.shape[0],1))
        return out

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the scores for each class for all of the data samples.

        Hint: this function is also used for prediction.

        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample

        Returns:
            Matrix of shape (N, C) where scores[i, c] is the score for class
                c on input X[i] outputted from the last layer of your network
        """
        self.outputs = {}
        # TODO: implement me. You'll want to store the output of each layer in
        # self.outputs as it will be used during back-propagation. You can use
        # the same keys as self.params. You can use functions like
        # self.linear, self.relu, and self.softmax in here.
        
        #Flatten the images -- The images seem to be already flattened.
        #Pass through the fully connected layers
        for layer in range(1,self.num_layers):
            X = self.outputs["linear"+str(layer)] = self.linear(self.params["W"+str(layer)],X,self.params["b"+str(layer)])
            X = self.outputs["relu"+str(layer)] = self.relu(X)
        #Pass through the softmax layer
        X = self.outputs["linear"+str(self.num_layers)] = self.linear(self.params["W"+str(self.num_layers)], X, self.params["b"+str(self.num_layers)])
        X = self.outputs["softmax"+str(self.num_layers)] = self.softmax(X)
        return X

    def backward(
        self, X: np.ndarray, y: np.ndarray, lr: float, reg: float = 0.0
    ) -> float:
        """Perform back-propagation and update the parameters using the
        gradients.

        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training sample
            y: Vector of training labels. y[i] is the label for X[i], and each
                y[i] is an integer in the range 0 <= y[i] < C
            lr: Learning rate
            reg: Regularization strength

        Returns:
            Total loss for this batch of training samples
        """
        
        self.gradients = {}
        loss = 0.0
        # TODO: implement me. You'll want to store the gradient of each layer
        # in self.gradients if you want to be able to debug your gradients
        # later. You can use the same keys as self.params. You can add
        # functions like self.linear_grad, self.relu_grad, and
        # self.softmax_grad if it helps organize your code.
        
        #backprop into softmax layer
        S  = self.outputs["softmax" + str(self.num_layers)]
        S1 = np.zeros(S.shape)
        S1[np.arange(y.size),y] = 1
        dX = - S1 + S  
        dW = self.gradients["W"+str(self.num_layers)] = np.matmul(self.outputs["relu"+str(self.num_layers-1)].T,dX)
        db = self.gradients["b"+str(self.num_layers)] = dX
        dX = np.matmul(dX,self.params["W"+str(self.num_layers)].T)

        self.outputs["relu"+str(0)] = X
        for layer in reversed(range(1,self.num_layers)):
            dR = self.outputs["relu"+str(layer)]
            dR = dX*(dR>0)
            dW = self.gradients["W"+str(layer)] = np.matmul(self.outputs["relu"+str(layer-1)].T,dR)
            db = self.gradients["b"+str(layer)] = dR
            dX = np.matmul(dR,self.params["W"+str(layer)].T)
            
            
        loss = -np.sum(np.log(S[np.arange(y.size),y])) 
        
        return loss
