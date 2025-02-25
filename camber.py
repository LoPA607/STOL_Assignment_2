import numpy as np
import matplotlib.pyplot as plt

def camber(x, y_func):
    y = y_func(x)
    return y

def camber_slope(x, y_func):
    slope = np.gradient(y_func(x), x)
    return slope

def naca(x, m, p, c, t):
    yt = 5*t*c*(0.2969*np.sqrt(x/c) - 0.1260*(x/c) - 0.3516*(x/c)**2 + 0.2843*(x/c)**3 - 0.1015*(x/c)**4)
    yc = np.zeros(len(x))
    dyc_dx = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] <= p*c:
            yc[i] = m/(p**2)*(2*p*(x[i]/c) - (x[i]/c)**2)
            dyc_dx[i] = 2*m/(p**2)*(p - x[i]/c)
        else:
            yc[i] = m/((1-p)**2)*(1 - 2*p + 2*p*(x[i]/c) - (x[i]/c)**2)
            dyc_dx[i] = 2*m/((1-p)**2)*(p - x[i]/c)
    return yt, yc, dyc_dx

def plot_naca(m, p, c, t):
    x = np.linspace(0, c, 1000)
    yt, yc, dyc_dx = naca(x, m, p, c, t)
    plt.plot(x, yt, label='Thickness')
    plt.plot(x, yc, label='Camber')
    plt.plot(x, dyc_dx, label='Camber Slope')
    plt.legend()
    plt.show()


