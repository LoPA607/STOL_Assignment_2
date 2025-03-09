import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
pi=np.pi
n=1000

def thin_airfoil(x, m, p):
    yc = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] <= p:
            yc[i] = m / p**2 * (2 * p * x[i] - x[i]**2)
        else:
            yc[i] = m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x[i] - x[i]**2)
    return yc

def camber_slope(x, m, p):

    if 0 <= x <= p:
        return m / p**2 * (2 * p  - 2 * x)
    else:
        return m / (1 - p)**2 * (2 * p - 2 * x)
    

def angle_from_x(x):
    n=len(x)
    theta=np.zeros(n)
    d_theta=np.zeros(n)
    for i in range(n):
        theta[i]=np.arccos(1-2*x[i])
    for i in range(n-1):
        d_theta[i]=theta[i+1]-theta[i]
    d_theta[n-1]=3e-14
    return theta,d_theta


def A_O(x,alpha,s):
    sum=0
    n=len(x)
    theta,d_theta=angle_from_x(x)
    for i in range(n):
        sum+=s[i]*d_theta[i]
    ans=alpha-(sum/pi)
    return ans

def An(x,alpha,s):
    theta,dtheta = angle_from_x(x)
    n=len(x)
    Ao = A_O(x,alpha,s)
    A = np.zeros(n+1)
    for i in range(n+1):
        if i == 0:
            A[i] = Ao
        else:
            for j in range(n):
                A[i] = A[i] + s[j]*dtheta[j]*np.cos((i)*theta[j])
            A[i] = (2*A[i])/np.pi

    return A

def cl_func(x,alpha,s):
  A=An(x,alpha,s)
  cl=pi*(2*A[0]+A[1])
  return cl

def cm_func(x,alpha,s):
    A=An(x,alpha,s)
    cm=-pi*(A[0]+A[1]-(A[2]/2))/2
    return cm



def plot_camber(x, yc):
    fig, ax = plt.subplots()
    ax.plot(x, yc, label='Camber Line')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 0.1)
    ax.set_title('Airfoil Camber Line')
    ax.legend()
    ax.grid()
    return fig


def streamlines(vel,A,x_values,y_values,alpha):
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

    x_grid = np.linspace(-1.5, 2.5, 20)
    y_grid = np.linspace(-1.5, 1.5, 20)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

    v_x = np.zeros((20,20))
    v_y = np.zeros((20,20))
    r_x = np.zeros(n)
    r_y = np.zeros(n)
    r = np.zeros(n)
    ds = np.zeros(n)
    for i in range(20):
        for k in range(20):
            for j in range(n):
                r_x[j] = -x_mesh[k][i] + x_values[j]
                r_y[j] = y_mesh[k][i] - y_values[j]
                r[j] =np.sqrt(r_x[j]*r_x[j] + r_y[j]*r_y[j])
                ds[j] = np.sqrt(1+camber_slope(x_values[j],m,p)**2)*(dx[j])

                v_x[k][i] = v_x[k][i] + (((r_y[j]) * gamma[j]*ds[j])/(2*np.pi*r[j]**2))
                v_y[k][i] = v_y[k][i] + (((r_x[j]) * gamma[j]*ds[j])/(2*np.pi*r[j]**2))

    u_x = vel*np.cos(alpha)
    u_y = vel*np.sin(alpha)

    for i in range(20):
        for j in range(20):
            v_x[j][i] = v_x[j][i] + u_x

    for i in range(20):
        for j in range(20):
            v_y[j][i] = v_y[j][i] + u_y
    
    fig, ax = plt.subplots()
    ax.quiver(x_mesh, y_mesh, v_x, v_y, scale=1000, width=0.002, color='black', pivot='mid')
    ax.plot(x_values, y_values, color='blue')
    ax.set_title('Vector Field around Airfoil')
    ax.set_xlabel('x-coordinate')
    ax.set_ylabel('y-coordinate')
    return fig

st.title('Thin Airfoil Analysis')

m = st.number_input('Maximum Camber (m)', min_value=0.01, max_value=0.1, value=0.04, step=0.01)
p = st.number_input('Position of Maximum Camber (p)', min_value=0.1, max_value=0.9, value=0.3, step=0.05)
vel = st.number_input('Freestream Velocity (m/s)', min_value=10, max_value=100, value=30, step=5)
alpha = np.radians(st.number_input('Angle of Attack (degrees)', min_value=-10, max_value=15, value=3, step=1))

#button to start computation
if st.button('Compute'):
    pass



x = np.linspace(0, 1, 1000)
yc = thin_airfoil(x, m, p)
s = np.zeros(n)
for i in range(n):
    s[i] = camber_slope(x[i], m, p)

cl = cl_func(x, alpha, s)
cm = cm_func(x, alpha, s)
A=An(x,alpha,s)
st.write(f'**Angle of Attack (α):** {np.degrees(alpha):.1f}°')

st.write(f'**Lift Coefficient (Cl):** {cl:.3f}')
st.write(f'**Moment Coefficient (Cm):** {cm:.3f}')

st.pyplot(plot_camber(x, yc))
st.pyplot(streamlines(vel, A, x, yc, alpha))
st.write('Streamlines plotted around the airfoil')
