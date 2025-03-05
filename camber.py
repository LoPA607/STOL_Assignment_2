import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

def thin_airfoil(x, m, p, c):
    # Initialize arrays for camber line (yc) and its slope (dyc_dx)
    yc = np.zeros(len(x))
    dyc_dx = np.zeros(len(x))
    
    for i in range(len(x)):
        if x[i] <= p * c:
            # Calculate camber line (yc) and its slope (dyc_dx) for the first part of the airfoil (x <= p*c)
            yc[i] = m / (p**2) * (2 * p * (x[i] / c) - (x[i] / c)**2)
            dyc_dx[i] = 2 * m / (p**2) * (p - x[i] / c)
        else:
            # Calculate camber line (yc) and its slope (dyc_dx) for the second part of the airfoil (x > p*c)
            yc[i] = m / ((1 - p)**2) * (1 - 2 * p + 2 * p * (x[i] / c) - (x[i] / c)**2)
            dyc_dx[i] = 2 * m / ((1 - p)**2) * (p - x[i] / c)
    
    return yc, dyc_dx

def plot_thin_airfoil(m, p, c):
    x = np.linspace(0, c, 1000)
    yc, dyc_dx = thin_airfoil(x, m, p, c)
    
    plt.figure()
    plt.plot(x, yc, label='Camber')
    plt.plot(x, dyc_dx, label='Camber Slope')
    plt.legend()
    plt.title('Thin Airfoil Camber and Camber Slope')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def vector_field(x, y_func, alpha):
    yc, dyc_dx = y_func(x)
    n = len(x)

    # Compute circulation using the Kutta-Joukowski theorem
    circulation = 2 * np.pi * alpha * simps(dyc_dx, x)
    print('Circulation:', circulation)

    # Define a coarser grid for better visibility
    X, Y = np.meshgrid(np.linspace(-0.5, 1.5, 10), np.linspace(-0.5, 0.5, 10))
    U = np.zeros_like(X)
    V = np.zeros_like(Y)

    epsilon = 1e-6  # Small value to prevent division by zero
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
          x_grid = X[i,j]
          y_grid = Y[i,j]
          min_dist = float('inf')
          for k in range(n):
            dist = np.sqrt((x_grid - x[k])**2 + (y_grid - yc[k])**2)
            min_dist = min(min_dist, dist)
          if min_dist > 0.05:
            for k in range(n):
              r = np.sqrt((x_grid - x[k])**2 + (y_grid - yc[k])**2 + epsilon)  # Prevent zero division
              theta = np.arctan2(y_grid - yc[k], x_grid - x[k])

              # Velocity components due to circulation
              dU = -circulation / (2 * np.pi * r) * np.sin(theta)
              dV = circulation / (2 * np.pi * r) * np.cos(theta)
              U[i,j] = U[i,j] + dU
              V[i,j] = V[i,j] + dV

    plt.figure()
    plt.quiver(X, Y, U, V, color='k', scale=0.05)
    plt.plot(x, yc, 'r', label='Camber Line')
    plt.title('Thin Airfoil Vector Field')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


# For a thin airfoil
# plot_thin_airfoil(0.06, 0.4, 1)

x = np.linspace(0, 1, 100)
# thin_airfoil_func = lambda x: thin_airfoil(x, 0.06, 0.4, 1)
# vector_field(x, thin_airfoil_func, 3)

#naca 2412
# thin_airfoil_func = lambda x: thin_airfoil(x, 0.02, 0.4, 1)
# vector_field(x, thin_airfoil_func, 3)

#naca 6409
thin_airfoil_func = lambda x: thin_airfoil(x, 0.06, 0.6, 1)
vector_field(x, thin_airfoil_func, 6)

# naca 4424
# thin_airfoil_func = lambda x: thin_airfoil(x, 0.04, 0.4, 1)
# vector_field(x, thin_airfoil_func, 12)

