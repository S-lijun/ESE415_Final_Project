from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from objective_function import f_quadratic, f_rastrigin ,f_rosenbrock, f_saddle

# Prepare mesh grid for plotting
x_vals = np.linspace(-2, 2, 100)
y_vals = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x_vals, y_vals)


# Define A and b for quadratic
A1 = np.array([[3, 1], [1, 2]])
b1 = np.array([1, 1])

# Define A for rastrigin
A2 = 10

# Define a and b for rosenbrock
a = 1
b_ = 100

def plot():
    
    Z_quadratic = np.zeros_like(X)
    Z_rastrigin = np.zeros_like(X)
    Z_saddle = np.zeros_like(X)
    Z_rosenbrock = np.zeros_like(X)


    # Populate the Z values
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x_vec = np.array([X[i, j], Y[i, j]])
            Z_quadratic[i, j] = f_quadratic(x_vec, A1, b1)
            Z_rastrigin[i, j] = f_rastrigin(x_vec, A2)
            Z_saddle[i, j] = f_saddle(x_vec)
            Z_rosenbrock[i, j] = f_rosenbrock(x_vec, a, b_)
    return Z_quadratic, Z_rastrigin, Z_saddle, Z_rosenbrock



# Plot all 4 in 3D
def plot_surface(Z, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    Z_quadratic, Z_rastrigin,Z_saddle,Z_rosenbrock = plot()

    plot_surface(Z_quadratic, "Quadratic Function")
    plot_surface(Z_rastrigin, "Rastrigin Function")
    plot_surface(Z_saddle, "Saddle Point Function")
    plot_surface(Z_rosenbrock, "Rosenbrock Function")
