#!/usr/bin/env python3

import numpy as np
import random
import argparse
import matplotlib.pyplot as plt
import time

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

def visualize(model, data, label, x, r):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    inside = label.flatten() == 1
    outside = ~inside
    ax = axes[0]
    ax.scatter(data[inside, 0], data[inside, 1], c="red", label="inside", s=20)
    ax.scatter(data[outside, 0], data[outside, 1], c="blue", label="outside", s=20)
    theta = np.linspace(0, 2*np.pi, 400)
    ax.plot(r*np.cos(theta), r*np.sin(theta), c="green", linewidth=2, label="true circle")
    ax.scatter(x[0,0], x[0,1], c="black", marker="x", s=100, label="input x")
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("Training data and true circle")
    ax = axes[1]
    prob = model.forward(data).flatten()
    sc = ax.scatter(data[:,0], data[:,1], c=prob, cmap="viridis", vmin=0, vmax=1, s=30)
    boundary = np.abs(prob - 0.5) < 0.05
    ax.scatter(data[boundary,0], data[boundary,1], c="white", s=40, label="estimated boundary")
    ax.plot(r*np.cos(theta), r*np.sin(theta), c="green", linewidth=2)
    ax.scatter(x[0,0], x[0,1], c="black", marker="x", s=100)
    ax.set_aspect("equal")
    ax.set_title("Predicted probability distribution")
    plt.colorbar(sc, ax=ax, label="P(inside)")
    plt.tight_layout()
    plt.show()

def generate_training_data(n = 300, r = 1.0, k = 2, seed = 0):
    random_generator = np.random.default_rng(seed)
    x = random_generator.uniform(- k * r, k * r, size = (n, 2))
    t = np.zeros((n, 1), dtype = np.float32)
    for i in range(n):
        xi = x[i, 0]
        yi = x[i, 1]
        # if xi**2 + yi**2 <= (r + 0.1)**2 and xi**2 + yi**2 >= (r - 0.1)**2:
        if xi**2 + yi**2 <= (r)**2:
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
    def backward(self, dLdy):
        self.dLdW = self.x.T @ dLdy
        self.dLdb = np.sum(dLdy, axis = 0, keepdims = True)
        dLdx = dLdy @ self.W.T
        return dLdx

class ReLU:
    def __init__(self):
        self.x = None
    def forward(self, x):
        self.x = x
        return np.where(x > 0, x, 0)
    def backward(self, dLdy):
        return np.where(self.x > 0, dLdy * 1, dLdy * 0)

class LeakyReLU:
    def __init__(self, alpha = 0.001):
        self.x = None
        self.alpha = alpha
    def forward(self, x):
        self.x = x
        return np.where(x > 0, x, self.alpha * x)
    def backward(self, dLdy):
        return np.where(self.x > 0, dLdy * 1, dLdy * self.alpha)

class Tanh:
    def __init__(self):
        self.y = None
    def forward(self, x):
        self.y = np.tanh(x)
        return self.y
    def backward(self, dLdy):
        return dLdy * (1 - self.y**2)

class Sigmoid:
    def __init__(self):
        self.y = None
    def forward(self, x):
        self.y = 1 / (1 + np.exp(-x))
        return self.y
    def backward(self, dLdy):
        # dL/dx
        return dLdy * self.y * (1 - self.y)

class Loss:
    def __init__(self):
        self.y = None
        self.t = None
    def forward(self, y, t):
        self.y = y
        self.t = t
        s = 1e-7    # nan 回避
        return - (self.t * np.log(self.y + s) + (1 - self.t) * np.log(1 - self.y + s))
    def backward(self):
        # dL/dy
        s = 1e-7    # nan 回避
        y = np.clip(self.y, s, 1 - s)
        dLdy = (y - self.t) / (y * (1 - y))
        return dLdy / y.shape[0]

class Model:
    def __init__(self, input_dimension, hidden_dimension, seed = 0):
        self.layers = [
                Affine(input_dimension, hidden_dimension, seed = seed),
                ReLU(),
                Affine(hidden_dimension, hidden_dimension, seed = seed + 1),
                ReLU(),
                Affine(hidden_dimension, 1, seed = seed + 5),
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
                layer.W = layer.W - alpha * layer.dLdW
                layer.b = layer.b - alpha * layer.dLdb

model = Model(2, 4, seed = int(time.time()))

# 学習率
alpha = args.alpha
epochs = args.epochs

data, label = generate_training_data(args.n, args.r, seed = 0)

# Train
for _ in range(epochs):
    for i in range(len(data)):
        d = data[i:i+1]
        l = label[i:i+1]
        model.forward(d)
        model.loss_calc(l)
        # print_vec("out", model.out)
        # print(f"{model.loss}")
        model.backward()
        model.step(alpha)

print(f"Loss: {model.loss}")

# Infer
x = np.array([[args.input[0], args.input[1]]])
v = model.forward(x)

print(f"Input: ({x[0][0]}, {x[0][1]})")
print(f"Result: {v}")

visualize(model, data, label, x, r = args.r)

