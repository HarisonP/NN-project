import parse_data as dataParser
import random
import numpy as np
from sknn.mlp import Regressor, Layer

import logging
logging.basicConfig()

def diff(first, second):
        return [item for item in first if item not in second]

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v

    return v / norm

random.seed(3)
train_data = dataParser.get_train_data()

test_set = random.sample(train_data, int(len(train_data) * 0.1));
train_data = diff(train_data, test_set)


nn = Regressor(
    layers=[
        Layer("Rectifier", units=2),
        Layer("Linear", units=1)],
    learning_rate=0.01,
    n_iter=500)

x_train = []
y_train = []

for row in train_data:
    x_train.append(row[0:2])
    y_train.append(row[2])

x_train = np.array(x_train)

for index, column in enumerate(x_train):
    x_train[index] = normalize(column)
    y_train[index] = x_train[index][0] + x_train[index][1]


# print(y_train)
x_train = np.array(x_train).reshape(len(x_train), 2)
y_train = np.array(y_train).reshape(len(y_train), 1)

# print(x_train, y_train)
nn.fit(x_train, y_train)

# print(a)

a = np.array([[0.3, 0.2]])
print(a)
print(nn.predict(a))

