#!/usr/bin/env python3

import numpy as np
import random
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "input",
            type = float,
            nargs = 2,
            help = "2D input: x y")
    parser.add_argument(
            "--epochs",
            type = int,
            default = 30,
            help = "epochs (default: 30)")
    parser.add_argument(
            "--alpha",
            type = float,
            default = 0.1,
            help = "学習率 (default: 0.1)")
    parser.add_argument(
            "-n",
            type = int,
            default = 100,
            help = "教師データの数 (default: 100)")
    parser.add_argument(
            "-r",
            type = float,
            default = 1,
            help = "円の半径 (default: 1)")
    return parser.parse_args()

args = get_args()

def print_vec(text, vec):
    print("*** " + text + " ***")
    print(vec)
    print("shape: " + str(vec.shape))
    print("")

def generate_training_data(n = 300, r = 1.0, k = 2, seed = 0):
    random_generator = np.random.default_rng(seed)
    x = random_generator.uniform(- k * r, k * r, size = (n, 2))
    t = np.zeros((n, 1), dtype = np.float32)
    for i in range(n):
        xi = x[i, 0]
        yi = x[i, 1]
        if xi**2 + yi**2 <= r**2:
            t[i, 0] = 1.0
        else:
            t[i, 0] = 0.0
    return x, t

class Affine:
    def __init__(self, input_dimension, output_dimension, seed = 0):
        random_generator = np.random.default_rng(seed)
        self.W = random_generator.standard_normal((input_dimension, output_dimension))
        self.b = np.zeros(output_dimension)
    def forward(self, x):
        self.x = x
        return x @ self.W + self.b
    def backward(self, dy):
        self.dW = self.x.T @ dy
        self.db = np.sum(dy, axis = 0)
        dx = dy @ self.W.T
        return dx

class ReLU:
    def __init__(self):
        self.x = None
    def forward(self, x):
        self.x = x
        return np.where(x > 0, x, 0)
    def backward(self, dy):
        return np.where(self.x > 0, dy, 0)

class Sigmoid:
    def __init__(self):
        self.y = None
    def forward(self, x):
        self.y = 1 / (1 + np.exp(-x))
        return self.y
    def backward(self, dy):
        return dy * self.y * ( 1 - self.y )

class Loss:
    def __init__(self):
        self.y = None
        self.t = None
    def forward(self, y, t):
        self.y = y
        self.t = t
        s = 1e-7
        return - (self.t * np.log(self.y + s) + (1 - self.t) * np.log(1 - self.y + s))
    def backward(self):
        return (self.y - self.t) / self.y.shape[0]

class Model:
    def __init__(self, input_dimension, hidden_dimension, seed = 0):
        self.layers = [
                Affine(input_dimension, hidden_dimension, seed = seed),
                ReLU(),
                Affine(hidden_dimension, 1, seed = seed),
                Sigmoid(),
                ]
        self.loss_function = Loss()

    def forward(self, x):
        v = x
        for layer in self.layers:
            v = layer.forward(v)
        self.out = v
        return v

    def loss_calc(self, t):
        self.loss = self.loss_function.forward(self.out, t)
        return self.loss

    def backward(self):
        ref = self.loss_function.backward()
        for layer in reversed(self.layers):
            ref = layer.backward(ref)
        return ref

    def step(self, alpha):
        for layer in self.layers:
            if isinstance(layer, Affine):
                layer.W = layer.W - alpha * layer.dW
                layer.b = layer.b - alpha * layer.db

model = Model(2, 4, seed = 0)

# 学習率
alpha = args.alpha
epochs = args.epochs

data, label = generate_training_data(args.n, args.r, seed = 0)

# Train
for _ in range(epochs):
    model.forward(data)
    model.loss_calc(label)
    # print_vec("out", model.out)
    # print(f"{model.loss}")
    model.backward()
    model.step(alpha)

# Infer
x = np.array([[args.input[0], args.input[1]]])
v = model.forward(x)
print_vec("RESULT", v)

