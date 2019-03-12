import sys
import csv
import random
from Matrix import Matrix
from NeuralNetwork import NeuralNetwork

inputNodes = 784
outputNodes = 10

nn = NeuralNetwork(inputNodes, 10, 16, outputNodes)

conArr = []

inputArr = []
with open("mnist_train.csv") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        conArr.append(row)
print("csv done")

inputMaArr = []
# array with the targets as a single number (not what we want)
numTarArr = []
# array with a 1 at the numTar element (this is how we want to represent a number)
# this array has a fixed size of the output nodes
tarArr = [float] * outputNodes
# an array of tarArr arrays
tarArrArr = []

for i in range(len(conArr)):
    conArr[i] = list(map(int, conArr[i]))

    numTarArr.append(conArr[i][0])
    for j in range(0, outputNodes):
        tarArr[j] = 0

    # 1 represents 100% confidence that its the number that the nn 'thinks' it is, everything else is 0
    # representing 0% confidence that it is the number at the index of the array corresponding to the number
    # that that index represents
    # i.g [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] representes a 2 ( from 0 to 9)
    tarArr[numTarArr[i]] = 1
    print("tarArr :", tarArr)
    tarArrArr.append(tarArr)
    print("tarArrArr[i]: ", tarArrArr[i])

    tarArrArr[i] = Matrix.tomatrix(tarArrArr[i])
    tarArrArr[i] = Matrix.statranse(tarArrArr[i])

    # delete the target number from the original array, to use it as the input matrix
    del conArr[i][0]

    # from 0 to 255 are too big numbers, we change it from 0 to 1 not including 0 and 1
    for k in range(0, len(conArr[i])):
        conArr[i][k] = conArr[i][k] / 255
        conArr[i][k] = conArr[i][k] * 0.9
        conArr[i][k] = conArr[i][k] + 0.01

    inputMa = Matrix.tomatrix(conArr[i])
    inputMaArr.append(inputMa)

    print("Feedforward: ", nn.feedforward(inputMaArr[i]).values)
    ffOutput = nn.feedforward(inputMaArr[i])

    errors = NeuralNetwork.errors(tarArrArr[i], ffOutput)
    print("errors: ", errors.values)


# print(len(conArr[0]))
# print(conArr[0])
# del conArr[0][0]
# print(len(conArr[0]))
# print(conArr[0])


# backpropagation(finderrors(feedforward())) maybe something like this, it would give more structure to the code
# but the problem is retaining the values aross the functions
