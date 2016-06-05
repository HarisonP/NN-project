import parse_data as dataParser
import random
import numpy as np
from sknn.mlp import Regressor, Layer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import logging
logging.basicConfig()


def diff(first, second):
        return [item for item in first if item not in second]

random.seed(3)
train_data = dataParser.get_train_data()
test_set = random.sample(train_data, int(len(train_data) * 0.1));
train_data = diff(train_data, test_set)
x_train = []
y_train_sum = []
x_test = []

for row in train_data:
    x_train.append(row[0:2])
    # y_train.append(row[2])


for row in test_set:
    x_test.append(row[0:2])

x_test = np.array(x_test)
x_train = np.array(x_train)

def train_for_sum(y_train):
    features_scaler = MinMaxScaler()
    for pair in features_scaler.fit_transform(x_train):
        y_train.append(pair[0] + pair[1])

    y_train = np.array(y_train).reshape(len(y_train), 1)

    pipeline = Pipeline([
            ('min/max scaler', MinMaxScaler(feature_range=(-1.0, 1.0), )),
            ('neural network', Regressor(
                                    layers=[
                                        Layer("Rectifier", units=2),
                                        Layer("Linear", units=1)],
                                    learning_rate=0.01,
                                    verbose=True,
                                    n_iter=1000))])
    pipeline.fit(x_train, y_train)

    return (pipeline, features_scaler)


def test_pipline(pipeline, scaler, test_data):
    result = pipeline.predict(test_data)
    real_results = [row[0] + row[1] for row in scaler.transform(test_data)]
    return mean_squared_error(real_results, result)


pipeline_sum, scaler = train_for_sum(y_train_sum)
print(test_pipline(pipeline_sum, scaler, x_test))
# BEST: 2.0669447007e-30