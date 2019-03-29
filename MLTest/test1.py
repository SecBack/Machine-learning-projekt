import sys
import random
from test3 import Matrix
from test2 import NeuralNetwork

nn = NeuralNetwork(2, 200, 1)

traning_data = [
    {"inputs": [0, 1], "targets": [1]},
    {"inputs": [1, 0], "targets": [1]},
    {"inputs": [0, 0], "targets": [0]},
    {"inputs": [1, 1], "targets": [0]},
]

arrinput = []
arrtar = []

for data in traning_data:
    arrinput.append(data["inputs"])
    arrtar.append(data["targets"])

inputmaarr = []
inputtararr = []

for i in range(0, len(arrtar)):
    inputmaarr.append(Matrix.tomatrix(arrinput[i]))
    inputtararr.append(Matrix.tomatrix(arrtar[i]))


q = Matrix.tomatrix([1, 0])
w = Matrix.tomatrix([0, 1])
e = Matrix.tomatrix([1, 1])
r = Matrix.tomatrix([0, 0])

for i in range(0, 100):
    temparr = list(zip(inputmaarr, inputtararr))
    random.shuffle(temparr)
    inputmaarr, inputtararr = zip(*temparr)

    for j in range(0, len(inputmaarr)):
        nn.train(inputmaarr[j], inputtararr[j])

    j = 0


print(
    "1,0: ",
    nn.feedforward(q).values,
    "0,1: ",
    nn.feedforward(w).values,
    "1,1: ",
    nn.feedforward(e).values,
    "0,0: ",
    nn.feedforward(r).values,
)
