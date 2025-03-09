import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

n = 1000
pi = np.pi

def user_def_camber_slope(x, custom_camber, epsilon):
    y1 = custom_camber(x + epsilon)
    y2 = custom_camber(x - epsilon)
    slope = (y1 - y2) / (2 * epsilon)
    return slope

def plot_user_camber_slope(x, user_slope):
    fig, ax = plt.subplots()
    ax.plot(x, user_slope, label='Camber')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Thin Airfoil Camber')
    ax.legend()
    ax.grid()
    return fig

def angle_from_x(x):
    n = len(x)
    theta = np.zeros(n)
    d_theta = np.zeros(n)
    for i in range(n):
        theta[i] = np.arccos(1 - 2 * x[i])
    for i in range(n - 1):
        d_theta[i] = theta[i + 1] - theta[i]
    d_theta[n - 1] = 3e-14
    return theta, d_theta

def A_O(x, alpha, s):
    sum = 0
    n = len(x)
    theta, d_theta = angle_from_x(x)
    for i in range(n):
        sum += s[i] * d_theta[i]
    ans = alpha - (sum / pi)
    return ans

def An(x, alpha, s):
    theta, dtheta = angle_from_x(x)
    n = len(x)
    Ao = A_O(x, alpha, s)
    A = np.zeros(n + 1)
    for i in range(n + 1):
        if i == 0:
            A[i] = Ao
        else:
            for j in range(n):
                A[i] += s[j] * dtheta[j] * np.cos(i * theta[j])
            A[i] = (2 * A[i]) / pi
    return A

def generate_camber_points(x_values, camber_function):
    x_coordinates = np.array(x_values)
    y_coordinates = np.array([camber_function(x) for x in x_values])
    return x_coordinates, y_coordinates

o=20
def streamlines_user(vel,A,x_values,y_values,custom_camber,alpha):
    n = len(x_values)  # Get the size of x_values
    dx = np.zeros(n)
    for i in range(n-1):
        dx[i] = x_values[i+1] - x_values[i]
    dx[n-1] = 0.001
    # Assuming theta is defined elsewhere and takes x_values as input
    theta_values, dtheta = angle_from_x(x_values)
    An_sum = np.zeros(n+1)

    for i in range(n):
        for j in range(n+1):
            if j == 0:
                continue
            else:
                An_sum[i] = An_sum[i] + A[j]*np.sin((j)*theta_values[i])

    gamma = np.zeros(n)

    for i in range(n):
        if np.sin(theta_values[i]) == 0:
            continue

        else:
            gamma[i] = 2*vel*(A[0]*(1+np.cos(theta_values[i]))/np.sin(theta_values[i]) + An_sum[i])

    gamma[0] = gamma[n-1]

    x_box = np.linspace(-1.5, 2.5, o)
    y_box = np.linspace(-1.5, 1.5, o)
    x_mesh, y_mesh = np.meshgrid(x_box, y_box)

    v_x = np.zeros((o,o))
    v_y = np.zeros((o,o))
    r_x = np.zeros(n)
    r_y = np.zeros(n)
    r = np.zeros(n)
    ds = np.zeros(n)
    for i in range(o):
        for k in range(o):
            for j in range(n):
                r_x[j] = -x_mesh[k][i] + x_values[j]
                r_y[j] = y_mesh[k][i] -y_values[j]
                r[j] =np.sqrt(r_x[j]*r_x[j] + r_y[j]*r_y[j])
                ds[j] = np.sqrt(1+user_def_camber_slope(x_values[j], custom_camber,1e-6)**2)*(dx[j])

                v_x[k][i] = v_x[k][i] + (((r_y[j]) * gamma[j]*ds[j])/(2*np.pi*r[j]**2))
                v_y[k][i] = v_y[k][i] + (((r_x[j]) * gamma[j]*ds[j])/(2*np.pi*r[j]**2))

    u_x = vel*np.cos(alpha)
    u_y = vel*np.sin(alpha)

    for i in range(o):
        for j in range(o):
            v_x[j][i] = v_x[j][i] + u_x

    for i in range(o):
        for j in range(o):
            v_y[j][i] = v_y[j][i] + u_y
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.quiver(x_mesh, y_mesh, v_x, v_y, scale=1000, width=0.002, color='black', pivot='mid')
    ax.plot(x_values, y_values, color='blue')
    ax.set_title('Vector Field around Airfoil')
    ax.set_xlabel('x-coordinate')
    ax.set_ylabel('y-coordinate')
    return fig

# Streamlit interface
st.title("Thin Airfoil Analysis")
st.sidebar.header("Custom Camber Line Inputs")

custom_equation = st.sidebar.text_input("Enter custom camber line (in terms of 'x')", "0.1*x*(1-x)")
alpha = np.radians(st.sidebar.slider('Angle of Attack (degrees)', -10, 15, 3, 1))
vel = st.sidebar.slider('Freestream Velocity (m/s)', 10, 100, 30, 5)

if st.sidebar.button("Plot Camber Slope and Streamlines"):
    custom_camber = lambda x: eval(custom_equation, {'x': x, 'np': np})
    x = np.linspace(0, 1, n)
    user_slope = user_def_camber_slope(x, custom_camber, 1e-6)
    fig_camber = plot_user_camber_slope(x, user_slope)
    st.pyplot(fig_camber)
    
    x_values, y_values = generate_camber_points(x, custom_camber)
    A = An(x, alpha, user_slope)
    fig_streamlines = streamlines_user(vel, A, x_values, y_values, custom_camber, alpha)
    st.pyplot(fig_streamlines)