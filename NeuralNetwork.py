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
        self.hidLayerWeiArr = []
        for i in range(0, (self.hidden_layers) - 1):
            self.hidLayerWei = Matrix(self.hidden_nodes, self.hidden_nodes)
            self.hidLayerWeiArr.append(self.hidLayerWei)
            self.hidLayerWeiArr[i].randomize()

        # create weights from the last hidden layer, to the output layer
        self.weightsHo = Matrix(self.output_nodes, self.hidden_nodes)
        self.weightsHo.randomize()

        # create biases for the hidden layer
        self.bias_h_arr = []
        for i in range(self.hidden_layers):
            hidLayerBi = Matrix(self.hidden_nodes, 1)
            self.bias_h_arr.append(hidLayerBi)
            self.bias_h_arr[i].randomize()

        self.bias_o = Matrix(self.output_nodes, 1)
        self.bias_o.randomize()

    def feedforward(self, input):
        self.input = input
        input = Matrix.statranse(input)

        # feedforward from input to hidden layer
        self.intohid = Matrix.multiply(self.weightsIh, input)
        self.intohid.add(self.bias_h_arr[0])
        self.intohid.map(self.intohid, sigmoid)

        # feedforward through the array of hidden layers
        self.hiddenArr = []
        self.hiddenArr.append(self.intohid)

        for i in range(0, len(self.hidLayerWeiArr)):
            hidden = Matrix.multiply(self.hidLayerWeiArr[i], self.hiddenArr[i])
            self.hiddenArr.append(hidden)
            self.hiddenArr[i + 1].add(self.bias_h_arr[i + 1])
            self.hiddenArr[i + 1].map(self.hiddenArr[i + 1], sigmoid)

        # feedforward from hidden to output layer
        output = Matrix.multiply(self.weightsHo, self.hiddenArr[-1])
        output.add(self.bias_o)
        output.map(output, sigmoid)

        return output

    @staticmethod
    def backprop(self, targets, ffOutput):
        # calculate output errors
        outputErrors = Matrix.subtract(targets, ffOutput)

        # calculate gradient for the output layer
        outGrad = Matrix.stamap(ffOutput, divsigmoid)
        outGrad = Matrix.elemulti(outGrad, outputErrors)  # 10x1 * 10x1
        outGrad.scale(self.learning_rate)

        # calculate deltas for the weights that go from last hidden layer to the output layer
        hiddenT = Matrix.statranse(self.hiddenArr[-1])
        weightHoDeltas = Matrix.multiply(outGrad, hiddenT)

        # adjust the weights by its deltas
        self.weightsHo.add(weightHoDeltas)

        # adjust biases by its deltas (which is just the gradients)
        self.bias_o.add(outGrad)

        # calculate gradient for hidden layer
        # notice index -i fetches the elements in the array from behind
        hidLayErrArr = []
        hidLayErrArr.append(outputErrors)
        self.hidLayerWeiArr.append(self.weightsHo)
        for i in range(1, self.hidden_layers):
            # find errors in the hidden layers, layer by layer
            weiHidLay = self.hidLayerWeiArr[-i]
            weiHidLayT = Matrix.statranse(weiHidLay)
            hidLayErr = Matrix.multiply(weiHidLayT, hidLayErrArr[i - 1])  # i-1
            hidLayErrArr.append(hidLayErr)

            # calculate the gradients for the hidden layers
            hidGrad = Matrix.stamap(self.hiddenArr[-i], divsigmoid)
            hidGrad = Matrix.elemulti(hidGrad, hidLayErrArr[-i])
            hidGrad.scale(self.learning_rate)

            # calcualte the deltas for the hidden layers
            hidddenT = Matrix.statranse(self.hiddenArr[(-i) - 1])
            weiHidLayDel = Matrix.multiply(hidGrad, hidddenT)

            # adjust the weights in the hidden layer with its delta
            self.hidLayerWeiArr[(-i) - 1].add(weiHidLayDel)

            # adjust the bias for each hidden neuron, with its deltas, with is the same as the gradient for the neuron
            self.bias_h_arr[-i].add(hidGrad)

        # find errors in the fist layer in the hidden layers
        weiHidLay = self.hidLayerWeiArr[0]
        weiHidLayT = Matrix.statranse(weiHidLay)
        intoHidErr = Matrix.multiply(weiHidLayT, hidLayErrArr[-1])  # i-1
        # find out which layer the loop above ends at first layer in the hidden layer, or the one after that

        # calculate gradient for the first layer in the hidden layer
        intoHidGrad = Matrix.stamap(self.hiddenArr[0], divsigmoid)
        intoHidGrad = Matrix.elemulti(intoHidGrad, intoHidErr)
        intoHidGrad.scale(self.learning_rate)

        # calculate deltas for the weights in the first hidden layer
        inputT = self.input  # is alerady transposed
        weightIhDeltas = Matrix.multiply(intoHidGrad, inputT)

        # adjust the weights by its deltas
        self.weightsIh.add(weightIhDeltas)

        # adjust biases by its deltas (which is just the gradients)
        self.bias_h_arr[0].add(intoHidGrad)

        del self.hidLayerWeiArr[-1]