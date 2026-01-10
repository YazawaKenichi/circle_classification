#!/usr/bin/env python3

import math
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

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def dsigmoid(y):
    return y * (1 - y)

class Affine:
    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.x = None

    def forward(self, x):
        self.x = x
        return self.w[0] * x[0] + self.w[1] * x[1] + self.b

    def backward(self, dL_dz):
        dL_dw = [dL_dz * self.x[0], dL_dz * self.x[1]]
        dL_db = dL_dz * 1
        return dL_dw, dL_db

    def update_parameter(self, dL_dy):
        dL_dw = [dL_dy * self.x[0], dL_dy * self.x[1]]
        self.w[0] = self.w[0] - self.alpha * dL_dw[0]
        self.w[1] = self.w[1] - self.alpha * dL_dw[1]
        self.b = dL_dy * 1

class Sigmoid:
    def __init__(self):
        self.y = None

    def forward(self, y):
        self.y = sigmoid(y)
        return self.y

    def backward(self, dL_dy):
        return dL_dy * dsigmoid(self.y)

def loss(z, t):
    return 1 / 2 * (z - t)**2

def dloss(z, t):
    return z - t

# 教師データの数
N = args.n
# 半径
R = args.r

# 入力
X = []
# ラベル
T = []

# 教師データの作成
for _ in range(N):
    p = random.uniform(-R, R)
    q = random.uniform(-R, R)
    X.append([p, q])
    T.append(1 if p**2 + q**2 <= R**2 else 0)
    #print(f"Data: {X[-1]} {T[-1]}")

w = [random.uniform(-R, R), random.uniform(-R, R)]
b = random.uniform(-R, R)

# 学習率
alpha = args.alpha
epochs = args.epochs

affine = Affine(w, b)
activation = Sigmoid()

# 学習
for epoch in range(epochs):
    for x, t in zip(X, T):
        # 順方向
        z = affine.forward(x)
        y = activation.forward(z)
        L = loss(y, t)
        # 逆伝搬
        dL_dy = dloss(y, t)
        dL_dz = activation.backward(dL_dy)
        dL_dw, dL_db = affine.backward(dL_dz)
        # パラメータの調整
        affine.w[0] = affine.w[0] - alpha * dL_dw[0]
        affine.w[1] = affine.w[1] - alpha * dL_dw[1]
        affine.b    = affine.b    - alpha * dL_db
        #print(f"Loss: {L}")

print(f"Train: {w}, {b}")

x = args.input
z = affine.forward(x)
y = activation.forward(z)
print(f"Infer: {y}")

