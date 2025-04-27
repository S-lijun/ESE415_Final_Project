import numpy as np

def generate_quadratic_params():
    # Random symmetric positive definite matrix A
    Q = np.random.randn(2, 2)
    A = Q.T @ Q + np.eye(2)  # ensure positive definiteness
    b = np.random.normal(loc=0, scale=6, size=2)
    return A, b

def generate_rastrigin_params():
    # A ~ N(10, 1)
    A = np.random.normal(loc=10, scale=6)
    return A,

def generate_saddle_params():
    # No parameters
    return ()

def generate_rosenbrock_params():
    a = np.random.normal(loc=1.0, scale=0.3)
    b = np.random.normal(loc=100.0, scale=20)
    return a, b
