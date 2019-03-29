from Matrix import Matrix
from math import exp


def sigmoid(self):
    for i in range(0, self.rows):
        for j in range(0, self.cols):
            self.values[i][j] = 1 / (1 + exp(-(self.values[i][j])))


def divsigmoid(self):
    # takes in a matirx that has been passed through the sigmoid function, then changes
    # that to the derivative of sigmoid
    return self * (1 - self)


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_layers, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_layers = hidden_layers
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = 0.1

        # input to hidden
        self.weightsIh = Matrix(self.hidden_nodes, self.input_nodes)
        self.weightsIh.randomize()

        # creates array of matricies corresponding to each hidden layer
        # -1 because i always takes n-1 things to connect n things in a line
        # hidLayerWeiArr
        self.weiArr = []
        for i in range(0, (self.hidden_layers) - 1):
            self.wei = Matrix(self.hidden_nodes, self.hidden_nodes)
            self.weiArr.append(self.wei)
            self.weiArr[i].randomize()

        # create weights from the last hidden layer, to the output layer
        self.weightsHo = Matrix(self.output_nodes, self.hidden_nodes)
        self.weightsHo.randomize()

        # adding input weights and outputweights to the weiArr
        self.weiArr.insert(0, self.weightsIh)
        self.weiArr.append(self.weightsHo)

        # create biases for the hidden layer
        self.biasArr = []
        for i in range(self.hidden_layers):
            hidLayerBi = Matrix(self.hidden_nodes, 1)
            self.biasArr.append(hidLayerBi)
            self.biasArr[i].randomize()

        self.bias_o = Matrix(self.output_nodes, 1)
        self.bias_o.randomize()

        # add output bias to biasArr
        self.biasArr.append(self.bias_o)

    def feedforward(self, input):
        # transpose the input
        self.input = input
        input = Matrix.statranse(input)

        # add the input to the array of nodes
        self.nodeArr = []
        self.nodeArr.append(input)

        # feedforward: node multiplied by weight, add bias to that then sigmoid the result, all that in a loop
        for i in range(0, len(self.weiArr)):
            node = Matrix.multiply(self.weiArr[i], self.nodeArr[i])
            self.nodeArr.append(node)
            self.nodeArr[i + 1].add(self.biasArr[i])
            self.nodeArr[i + 1].map(self.nodeArr[i + 1], sigmoid)

        # return the result of the last node, which is the output layer
        return self.nodeArr[-1]

    @staticmethod
    def backprop(self, targets, ffOutput):
        # calculate output errors
        outputErrors = Matrix.subtract(targets, ffOutput)

        # calculate gradient for the output layer
        outGrad = Matrix.stamap(ffOutput, divsigmoid)
        outGrad = Matrix.elemulti(outGrad, outputErrors)
        outGrad.scale(self.learning_rate)

        # calculate deltas for the weights that go from last hidden layer to the output layer
        hiddenT = Matrix.statranse(self.nodeArr[-2])
        weightHoDeltas = Matrix.multiply(outGrad, hiddenT)

        # adjust the weights by its deltas
        self.weiArr[-1].add(weightHoDeltas)

        # adjust biases by its deltas (which is just the gradients)
        self.biasArr[-1].add(outGrad)

        # calculate gradient for hidden layer
        # notice index -i fetches the elements in the array from behind
        nodeErrArr = []
        nodeErrArr.append(outputErrors)

        for i in range(1, len(self.weiArr)):
            # find errors in the hidden layers, layer by layer
            weiHidLay = self.weiArr[-i]
            weiHidLayT = Matrix.statranse(weiHidLay)
            hidLayErr = Matrix.multiply(weiHidLayT, nodeErrArr[i - 1])
            nodeErrArr.append(hidLayErr)

            # calculate the gradients for the hidden layers
            hidGrad = Matrix.stamap(self.nodeArr[(-i) - 1], divsigmoid)
            hidGrad = Matrix.elemulti(hidGrad, nodeErrArr[-1])
            hidGrad.scale(self.learning_rate)

            # calcualte the deltas for the hidden layers
            hidddenT = Matrix.statranse(self.nodeArr[(-i) - 2])
            weiHidLayDel = Matrix.multiply(hidGrad, hidddenT)

            # adjust the weights in the hidden layer with its delta
            self.weiArr[(-i) - 1].add(weiHidLayDel)

            # adjust the bias for each hidden neuron, with its deltas, which is the same as the gradient for the neuron
            self.biasArr[(-i) - 1].add(hidGrad)