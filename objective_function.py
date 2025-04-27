import numpy as np
import pandas as pd


# In this project we only consider 2-dimensional input
# 1. Convex Quadratic Function: f(x) = x^T A x + b^T x

def f_quadratic(x, A, b):
    return x.T @ A @ x + b.T @ x

def grad_quadratic(x, A, b):
    return 2 * A @ x + b

# 2. Rastrigin Function: f(x) = 10n + sum(x_i^2 - 10cos(2pi x_i))
def f_rastrigin(x, A=10):
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def grad_rastrigin(x, A=10):
    return 2 * x + 2 * np.pi * A * np.sin(2 * np.pi * x)

# 3. Saddle Point Function: f(x, y) = x^2 - y^2
def f_saddle(x):
    return x[0]**2 - x[1]**2

def grad_saddle(x):
    return np.array([2 * x[0], -2 * x[1]])

# 4. Rosenbrock Function: f(x, y) = (a - x)^2 + b(y - x^2)^2
def f_rosenbrock(x, a=1, b=100):
    return (a - x[0])**2 + b * (x[1] - x[0]**2)**2

def grad_rosenbrock(x, a=1, b=100):
    dx = -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0]**2)
    dy = 2 * b * (x[1] - x[0]**2)
    return np.array([dx, dy])

