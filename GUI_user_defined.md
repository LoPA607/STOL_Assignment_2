# Thin Airfoil Analysis

This Streamlit application provides an interactive interface to analyze the camber slope and vector field around a thin airfoil using custom camber line equations. Users can input a custom camber equation, adjust parameters, and visualize the results.

## Features
- Define a **custom camber line** as a function of `x`
- Set **angle of attack (alpha)** in degrees
- Adjust **freestream velocity**
- Plot **camber slope** for the given custom camber line
- Generate and visualize **streamlines (vector field) around the airfoil**

## Installation
To run the application, ensure you have Python installed along with the required dependencies:

```sh
pip install streamlit numpy matplotlib
```

## Usage
Run the application with:

```sh
streamlit run final2.py
```

## Input Parameters
- **Custom Camber Line**: A mathematical equation in terms of `x` (e.g., `0.1*x*(1-x)`).
- **Angle of Attack**: Selectable via a slider (-10 to 15 degrees).
- **Freestream Velocity**: Selectable via a slider (10 to 100 m/s).

## Output
- **Camber Slope Plot**: Displays the slope variation along the airfoil.
- **Vector Field Plot**: Shows the velocity field around the airfoil based on circulation theory.

## Code Structure
- `user_def_camber_slope(x, custom_camber, epsilon)`: Computes the slope of the custom camber line.
- `plot_user_camber_slope(x, user_slope)`: Generates a plot of the camber slope.
- `angle_from_x(x)`: Computes the theta distribution for panel method calculations.
- `A_O(x, alpha, s)`: Computes the `A0` coefficient for airfoil lift calculations.
- `An(x, alpha, s)`: Computes Fourier coefficients for the thin airfoil theory.
- `generate_camber_points(x_values, camber_function)`: Generates x and y points based on camber function.
- `streamlines_user(vel, A, x_values, y_values, custom_camber, alpha)`: Generates the velocity field around the airfoil.
- Streamlit UI elements for user interaction.

## Example
If a user inputs the camber function `0.1*x*(1-x)`, an angle of attack of `3°`, and a freestream velocity of `30 m/s`, the app will:
1. Compute the camber slope.
2. Display the camber slope plot.
3. Compute the velocity field and display the vector plot around the airfoil.


