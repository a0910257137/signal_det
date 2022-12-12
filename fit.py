import numpy as np
from matplotlib import pyplot as plt


def f(params, x):
    y0, k, b = params[0], params[1], params[2]
    return y0 * np.exp(-k * x) + b


def trapezoidal(params, x0, xn, sub_n):
    h = (xn - x0) / sub_n
    integration = f(params, x0) + f(params, xn)
    sub_b = np.linspace(1, sub_n, sub_n)
    k = x0 + sub_b * h
    trps = 2 * f(params, k) * h / 2
    integration = trps.sum()
    return integration


def X(params):
    integration = trapezoidal(params, 0, 2, 100)
    return integration


def cost_func(n, y, params):
    y_pred = f(params, n)
    cost = (y_pred - y)**2
    return cost


def grad(n, raw_y, params):
    y0, k, b = params[0], params[1], params[2]
    if k != 0:
        tau = 1 / k
        lower = (2 * tau + 1) / (2 * tau - 1)
        if lower > 0:
            ct_tau = 1 / (np.log(lower))
            k = 1 / ct_tau
            params[1] = k

    A, B, C = y0 + b, -k, k * b
    grad_A = -2 * (raw_y - A - B * X(params) - C * n)
    grad_B = -2 * (raw_y - A - B * X(params) - C * n) * X(params)
    grad_C = -2 * (raw_y - A - B * X(params) - C * n) * n
    return grad_A, grad_B, grad_C


# example
n = np.linspace(0, 5, 100)
raw_y = 6 * np.exp(-4 * n)

# initial parameters
params = [0.1, 0.1, 0]

bounds = {"y0": [0.01, 10], "k": [0.01, 10], "b": [0, 0.02]}
lr = 0.1
epochs = 100
costs = []
for k in range(epochs):
    cost = cost_func(n, raw_y, params)
    cost = np.sum(cost) / cost.shape[0]
    costs.append(cost)
    grad_A, grad_B, grad_C = grad(k, n, raw_y, params)
    grad_A = np.mean(grad_A)
    grad_B = np.mean(grad_B)
    grad_C = np.mean(grad_C)
    y0, k, b = params[0], params[1], params[2]
    A, B, C = y0 + b, -k, k * b

    A = A - lr * grad_A
    B = B - lr * grad_B
    C = C - lr * grad_C
    # decompose
    k = -B
    b = C / k
    y0 = A - b

    if y0 <= bounds['y0'][0]:
        y0 = 1
    elif y0 >= bounds['y0'][1]:
        y0 = 1

    if k <= bounds['k'][0]:
        k = 1

    elif k >= bounds['k'][1]:
        k = 1

    if b <= bounds['b'][0]:
        b = 0
    elif b >= bounds['b'][1]:
        b = 0
    params[0] = y0
    params[1] = k
    params[2] = b

costs = np.asarray(costs)
min_indxs = np.argmin(costs)
