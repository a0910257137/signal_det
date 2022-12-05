import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.optimize import curve_fit


def func(x, a, b, c):
    return a * np.exp(-b * x) + c


xdata = np.linspace(0, 4, 50)
y = func(xdata, 2.5, 1.3, 0.5)
rng = np.random.default_rng()
y_noise = 0.2 * rng.normal(size=xdata.size)
ydata = y + y_noise
popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [3., 1., 0.5]))


def bias_coef_update(m, b, X, Y, learning_rate):
    m_gradient = 0
    b_gradient = 0

    N = len(Y)

    # iterate over examples
    for idx in range(len(Y)):
        x = X[idx]
        y = Y[idx]

        # predict y with current bias and coefficient
        y_hat = (m * x) + b
        m_gradient += -(2 / N) * x * (y - y_hat)
        b_gradient += -(2 / N) * (y - y_hat)

    # use gradient with learning_rate to nudge bias and coefficient
    new_coef = m - (m_gradient * learning_rate)
    new_bias = b - (b_gradient * learning_rate)

    return new_coef, new_bias


def run(X, epoch_count=1000):
    # store output to plot later
    epochs = []
    costs = []

    m = 0
    b = 0
    learning_rate = 0.01
    for i in range(epoch_count):
        m, b = bias_coef_update(m, b, X, y, learning_rate)
        print(m, b)

        C = cost(b, m, x_y_pairs)

        epochs.append(i)
        costs.append(C)

    return epochs, costs, m, b


epochs, costs, m, b = run()


def find_theta(X, y):

    m = X.shape[0]  # Number of training examples.
    # Appending a cloumn of ones in X to add the bias term.
    X = np.append(X, np.ones((m, 1)), axis=1)
    # reshaping y to (m,1)
    y = y.reshape(m, 1)
    # The Normal Equation
    theta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

    return theta


def predict(X, theta):

    # Appending a cloumn of ones in X to add the bias term.
    X = np.append(X, np.ones((X.shape[0], 1)), axis=1)

    # preds is y_hat which is the dot product of X and theta.
    preds = np.dot(X, theta)

    return preds


fig = plt.figure(figsize=(8, 6))
t = np.linspace(0, 10, 100)[:, None]
y = 10 * np.exp(-5 * t) + 10
plt.plot(t, y, 'r-')
log_y = np.log(y)
theta = find_theta(t, log_y)

print(theta)
preds = predict(t, theta=theta)

plt.plot(t, log_y, 'b.')
plt.plot(t, preds, 'c-')
plt.grid()
plt.savefig('foo.png')