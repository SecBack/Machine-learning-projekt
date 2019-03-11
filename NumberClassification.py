import sys
import csv
import random
from Matrix import Matrix
from NeuralNetwork import NeuralNetwork

input = 784

nn = NeuralNetwork(input, 10, 16, 10)

conArr = []
tarArr = []
inputArr = []
with open("mnist_train_test.csv") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        conArr.append(row)
print("csv done")

inputMaArr = []
for i in range(len(conArr)):
    conArr[i] = list(map(int, conArr[i]))
    tarArr.append(conArr[i][0])
    del conArr[i][0]

    # from 0 to 255 are too big numbers, we change it from 0 to 1 not including 0 and 1
    for j in range(0, len(conArr[i])):
        conArr[i][j] = conArr[i][j] / 255
        conArr[i][j] = conArr[i][j] * 0.9
        conArr[i][j] = conArr[i][j] + 0.01

    inputMa = Matrix.tomatrix(conArr[i])
    inputMaArr.append(inputMa)

    print("Feedforward: ", nn.feedforward(inputMaArr[i]).values)
    #ffOutput = nn.feedforward(inputMaArr[i])

    # i only have 2 parameters, why do i nedd 3?????
    #errors = NeuralNetwork.errors(ffOutput, tarArr[i])


# print(len(conArr[0]))
# print(conArr[0])
# del conArr[0][0]
# print(len(conArr[0]))
# print(conArr[0])


# backpropagation(finderrors(feedforward())) maybe something like this, it would give more structure to the code
# but the problem is retaining the values aross the functions


# print(nn.feedforward(inputMa).values)


# traning_data = [
#     {"inputs": [0, 1], "targets": [1]},
#     {"inputs": [1, 0], "targets": [1]},
#     {"inputs": [0, 0], "targets": [0]},
#     {"inputs": [1, 1], "targets": [0]},
# ]

# arrinput = []
# arrtar = []

# for data in traning_data:
#     arrinput.append(data["inputs"])
#     arrtar.append(data["targets"])

# inputmaarr = []
# inputtararr = []

# for i in range(0, len(arrtar)):
#     inputmaarr.append(Matrix.tomatrix(arrinput[i]))
#     inputtararr.append(Matrix.tomatrix(arrtar[i]))


# q = Matrix.tomatrix([1, 0])
# w = Matrix.tomatrix([0, 1])
# e = Matrix.tomatrix([1, 1])
# r = Matrix.tomatrix([0, 0])

# for i in range(0, 100):
#     temparr = list(zip(inputmaarr, inputtararr))
#     random.shuffle(temparr)
#     inputmaarr, inputtararr = zip(*temparr)

#     for j in range(0, len(inputmaarr)):
#         nn.train(inputmaarr[j], inputtararr[j])

#     j = 0


# print(
#     "1,0: ",
#     nn.feedforward(q).values,
#     "0,1: ",
#     nn.feedforward(w).values,
#     "1,1: ",
#     nn.feedforward(e).values,
#     "0,0: ",
#     nn.feedforward(r).values,
# )
