import numpy as np
from scipy.optimize import minimize_scalar

# 1. Standard Gradient Descent (Fixed Step Size)
def gradient_descent(f, grad_f, x0, args=(), step_size=0.01, max_iters=1000, tol=1e-6):
    x = x0.copy()
    trajectory = [x.copy()]
    for _ in range(max_iters):
        grad = grad_f(x, *args)
        x_new = x - step_size * grad
        trajectory.append(x_new.copy())
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return x, trajectory

# 2. Gradient Descent with Exact Line Search (only works well for convex)
def exact_line_search(f, grad_f, x0, args=(), max_iters=1000, tol=1e-6):
    
    x = x0.copy()
    trajectory = [x.copy()]
    for _ in range(max_iters):
        d = -grad_f(x, *args)
        res = minimize_scalar(lambda alpha: f(x + alpha * d, *args))
        alpha = res.x
        x_new = x + alpha * d
        trajectory.append(x_new.copy())
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return x, trajectory

# 3. Backtracking Line Search
def backtracking_line_search(f, grad_f, x0, args=(), alpha=0.5, beta=0.8, max_iters=1000, tol=1e-6):
    x = x0.copy()
    trajectory = [x.copy()]
    for _ in range(max_iters):
        grad = grad_f(x, *args)
        t = 1
        while f(x - t * grad, *args) > f(x, *args) - alpha * t * np.linalg.norm(grad)**2:
            t *= beta
        x_new = x - t * grad
        trajectory.append(x_new.copy())
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return x, trajectory

# 4. Accelerated Gradient Method (Nesterov)
def accelerated_gradient(f, grad_f, x0, args=(), step_size=0.01, max_iters=1000, tol=1e-6):
    x = x0.copy()
    y = x0.copy()
    trajectory = [x.copy()]
    for t in range(1, max_iters+1):
        x_prev = x.copy()
        grad = grad_f(y, *args)
        x = y - step_size * grad
        beta = (t - 1) / (t + 2)
        y = x + beta * (x - x_prev)
        trajectory.append(x.copy())
        if np.linalg.norm(x - x_prev) < tol:
            break
    return x, trajectory

