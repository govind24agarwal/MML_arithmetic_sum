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

    def forward_propagate(self, inputs):
        """Method to compute forward propagationof thenetwork based on input signals
        Args:
            inputs (ndArray): Input Values
        Returns:
            activations (ndArray): Output Values
        """
        activations = inputs

        # ittiratethrough network layers
        for w in self.weights:
            # calculate  dot product
            net_inputs = np.dot(activations, w)
            # apply sigmoid function
            activations = self._sigmoid(net_inputs)

        return activations

    def _sigmoid(self, x):
        """Sigmoid activation function
        Args:
            x (float): value to be processed
        Returns:
            y (float): Output
        """

        y = 1.0/(1 + np.exp(-x))
        return y


if __name__ == "__main__":
    mlp = MLP()
    inputs = np.random.rand(mlp.num_of_inputs)
    output = mlp.forward_propagate(inputs)

    print("Network Inputs : {}".format(inputs))
    print("Network Outputs : {}".format(output))
