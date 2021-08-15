import numpy as np


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


if __name__ == "__main__":
    mlp = MLP(2, [5], 1)
    inputs = np.array([0.1, 0.2])
    target = np.array([0.3])
    output = mlp.forward_propagate(inputs)
    error = target - output
    mlp.back_propagate(error, True)
