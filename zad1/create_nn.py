import parse_data as dataParser
import random

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

from pybrain.structure import SigmoidLayer
from pybrain.structure import TanhLayer
from pybrain.structure import SoftmaxLayer


def diff(first, second):
        return [item for item in first if item not in second]

def train_net(data, net, target_index):
    ds = SupervisedDataSet(2, 1)
    for row in data:
        print(row[0:2])
        output = []

        output.append(row[target_index])
        print(output)
        ds.addSample(row[0:2], output)

    trainer = BackpropTrainer(net, ds,
                    learningrate=0.01,
                    lrdecay= 1.0,
                    momentum=0.3,
                    verbose=True,
                    batchlearning=False,
                    weightdecay=0.0)

    trainer.trainEpochs(epochs=100)

random.seed(3)
train_data = dataParser.get_train_data()
train_data

# test_set = random.sample(train_data, int(len(train_data) * 0.1));
# train_data = diff(train_data, test_set)

net = buildNetwork(2, 1, 1, hiddenclass=SigmoidLayer , bias=True)


train_net(train_data, net, 2)

print(net.activate([170, -3]))