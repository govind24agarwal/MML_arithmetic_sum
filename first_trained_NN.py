import numpy as np
from random import random


class MLP(object):
    """A multiplayer Perception class"""

    def __init__(self, num_of_inputs=3, hidden_layers=[4, 2], num_of_outputs=2):
        """Constructor of MLP class.
        It creates weight arrayfor weights of connects between nodes of layers.
        Args:
            num_of_inputs (int): Number of inputs
            hidden_layers (list): List ofnumber of nodes in each hidden layer
            num_of_outputs (int): Number of outputs expected
        """

        self.num_of_inputs = num_of_inputs
        self.hidden_layers = hidden_layers
        self.num_of_outputs = num_of_outputs

        # Combining no ofnodes in each layer in one array
        layers = [num_of_inputs] + hidden_layers + [num_of_outputs]

        # Creating weight 3d array that contains2d arrays of weights of connections between two consecutive layers
        weights = []
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1])
            weights.append(w)
        self.weights = weights

        # save activations per layer
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

        # save derivatives per layer
        derivatives = []
        for i in range(len(layers)-1):
            d = np.zeros((layers[i], layers[i+1]))
            derivatives.append(d)
        self.derivatives = derivatives

    def forward_propagate(self, inputs):
        """Method to compute forward propagationof thenetwork based on input signals
        Args:
            inputs (ndArray): Input Values
        Returns:
            activations (ndArray): Output Values
        """
        activations = inputs
        self.activations[0] = inputs

        # ittiratethrough network layers
        for i, w in enumerate(self.weights):
            # calculate  dot product
            net_inputs = np.dot(activations, w)
            # apply sigmoid function
            activations = self._sigmoid(net_inputs)
            # save the activations for back propogation
            self.activations[i+1] = activations
        return activations

    def back_propagate(self, error, verbose=False):
        """Backpropogate an errorsignal

        Args:
            error (ndarray): The error to backdrop

        Returns:
            error (ndarray): The final error of the input
        """
        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i+1]
            delta = error * self._sigmoid_derivative(activations)
            delta_reshaped = delta.reshape(delta.shape[0], -1).T
            current_activations = self.activations[i]
            current_activations = current_activations.reshape(
                current_activations.shape[0], -1)
            self.derivatives[i] = np.dot(current_activations, delta_reshaped)
            error = np.dot(delta, self.weights[i].T)

            if(verbose):
                print("Derivatives for W{}: {}".format(i, self.derivatives[i]))

        return error

    def gradient_descent(self, learning_rate=1):
        """Learns by descending the  gradient
        Args:
            learning_rate (float, optional): How fast to learn. Defaults to 1.
        """
        # update the weights  by stepping down the gradient
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * learning_rate

    def train(self, inputs, targets, epochs, learning_rate):
        """Trains model running forward propagation and back propagation

        Args:
            inputs (ndarray) : X
            targets (ndarray): Y
            epochs (int): Number of times we want to train the network for
            learning_rate (float): step to apply to gradient descent
        """
        for i in range(epochs):
            sum_error = 0
            for input, target in zip(inputs, targets):
                output = self.forward_propagate(input)
                error = target-output
                self.back_propagate(error)
                self.gradient_descent(learning_rate)
                sum_error += self._mse(target, output)
            print("Error at {} epoch : {}".format(sum_error/len(inputs), i+1))

    def _sigmoid(self, x):
        """Sigmoid activation function
        Args:
            x (float): value to be processed
        Returns:
            y (float): Output
        """

        y = 1.0/(1 + np.exp(-x))
        return y

    def _sigmoid_derivative(self, x):
        """Sigmoid derivative function
        Args:
            x (float): Value to be processed (Sigmoid of some a -> s[a])
        Returns:
            y (float): Output (Derivative of sigmoid of a -> s'[a])
        """
        return x * (1.0-x)

    def _mse(self, target, output):
        """Mean squared error loss function

        Args:
            target (ndarray): The ground trut
            output (ndarray): The predicted value

        Returns:
            (float): Output
        """
        return np.average((target-output)**2)


if __name__ == "__main__":

    # create a dataset to train a network for the sum operation
    items = np.array([[random()/2 for _ in range(2)] for _ in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in items])

    # create a Multilayer Perceptron with one hidden layer
    mlp = MLP(2, [5], 1)

    # train network
    mlp.train(items, targets, 50, 0.1)

    # create dummy data
    input = np.array([0.3, 0.1])
    target = np.array([0.4])

    # get a prediction
    output = mlp.forward_propagate(input)

    print()
    print(
        "Our network believes that {} + {} is equal to {}".format(input[0], input[1], output[0]))
