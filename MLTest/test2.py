from test3 import Matrix
from math import exp


def sigmoid(self):
    for i in range(0, self.rows):
        for j in range(0, self.cols):
            self.values[i][j] = 1 / (1 + exp(-(self.values[i][j])))


def divsigmoid(self):
    # takes in a matirx that has been passed through the sigmoid function, then changes that to the derivative function of sigmoid
    return self * (1 - self)


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = 0.5

        # create weight matricies
        self.weights_ih = Matrix(self.hidden_nodes, self.input_nodes)
        self.weights_ho = Matrix(self.output_nodes, self.hidden_nodes)
        self.weights_ih.randomize()
        self.weights_ho.randomize()

        # create bias Matricies
        self.bias_h = Matrix(self.hidden_nodes, 1)
        self.bias_o = Matrix(self.output_nodes, 1)
        self.bias_h.randomize()
        self.bias_o.randomize()

    def feedforward(self, input):
        self.input = input

        input = Matrix.statranse(input)

        # feedforward from input to hidden layer
        hidden = Matrix.multiply(self.weights_ih, input)
        hidden.add(self.bias_h)
        hidden.map(hidden, sigmoid)

        # feedforward from hidden to output layer
        output = Matrix.multiply(self.weights_ho, hidden)
        output.add(self.bias_o)
        output.map(output, sigmoid)

        return output

    def train(self, input, targets):

        # feedforward pasted
        # feedforward from input to hidden layer
        input = Matrix.statranse(input)
        print(
            "weights_ih rows: ",
            self.weights_ih.rows,
            "weights_ih cols: ",
            self.weights_ih.cols,
        )
        print("input rows: ", input.rows, "input cols: ", input.cols)
        hidden = Matrix.multiply(self.weights_ih, input)
        print("hidden rows: ", hidden.rows, "hidden cols: ", hidden.cols, "\n")

        hidden.add(self.bias_h)
        hidden.map(hidden, sigmoid)

        # feedforward from hidden to output layer
        print(
            "weights_ho rows: ",
            self.weights_ho.rows,
            "weights_ho cols: ",
            self.weights_ho.cols,
        )
        print("hidden rows: ", hidden.rows, "hidden cols: ", hidden.cols)
        outputs = Matrix.multiply(self.weights_ho, hidden)
        print("outputs rows: ", outputs.rows, "outputs cols: ", outputs.cols, "\n")

        outputs.add(self.bias_o)
        outputs.map(outputs, sigmoid)

        # finds errors in output
        output_errors = Matrix.subtract(targets, outputs)

        # calculate gradient
        gradients = Matrix.stamap(outputs, divsigmoid)
        print("grad rows: ", gradients.rows, "grad cols: ", gradients.cols)
        print("outputsErr rows: ", outputs.rows, "outputsErr cols: ", outputs.cols)
        gradients = Matrix.multiply(gradients, output_errors)
        print(
            "gradients after rows: ",
            gradients.rows,
            "gradients after cols: ",
            gradients.cols,
            "\n",
        )
        gradients.scale(self.learning_rate)

        # calcualte deltaes
        hidden_T = hidden.statranse(hidden)
        # weight_ho_deltas = Matrix.multiply(gradients, hidden_T)
        print("gradients rows: ", gradients.rows, "gradients cols: ", gradients.cols)
        print("hidden_T rows: ", hidden_T.rows, "hidden_T cols: ", hidden_T.cols)
        weight_ho_deltas = Matrix.multiply(gradients, hidden_T)
        print(
            "weight_ho_deltas rows: ",
            weight_ho_deltas.rows,
            "weight_ho_deltas cols: ",
            weight_ho_deltas.cols,
            "\n",
        )

        # adjust the weights by deltas
        self.weights_ho.add(weight_ho_deltas)
        # adjust biases by its deltas (which is just the gradients)
        self.bias_o.add(gradients)

        # find errors in hidden
        who_t = self.weights_ho.statranse(self.weights_ho)
        hidden_errors = Matrix.multiply(who_t, output_errors)

        # calculate hidden gradients
        hidden_gradients = Matrix.stamap(hidden, divsigmoid)
        hidden_gradients = Matrix.elemulti(hidden_gradients, hidden_errors)
        hidden_gradients.scale(self.learning_rate)

        # calculate input to hidden deltas
        inputs_T = input.statranse(input)
        weight_ih_deltas = Matrix.multiply(hidden_gradients, inputs_T)
        self.weights_ih.add(weight_ih_deltas)

        # adjust the weights by deltas
        self.weights_ih.add(weight_ih_deltas)
        # adjust biases by its deltas (which is just the gradients)
        self.bias_h.add(hidden_gradients)
