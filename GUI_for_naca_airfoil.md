# GUI_for_naca_airfoil
## Thin Airfoil Analysis

This Streamlit application analyzes thin airfoils using the thin airfoil theory. It computes and visualizes the camber line, lift coefficient (Cl), moment coefficient (Cm), and streamlines around the airfoil for a given maximum camber, position of maximum camber, freestream velocity, and angle of attack.

## Features
- Compute the camber line of the airfoil based on the given parameters.
- Calculate aerodynamic coefficients:
  - **Lift Coefficient (Cl)**
  - **Moment Coefficient (Cm)**
- Display the airfoil camber line plot.
- Generate and visualize streamlines around the airfoil to show flow behavior.

## Dependencies
This application requires the following Python libraries:
```bash
pip install streamlit numpy matplotlib
```

## Usage
Run the application using the command:
```bash
streamlit run final.py
```

### Input Parameters
- **Maximum Camber (m):** Defines the maximum height of the camber line.
- **Position of Maximum Camber (p):** The x-location where the maximum camber occurs.
- **Freestream Velocity (m/s):** The velocity of air flowing over the airfoil.
- **Angle of Attack (degrees):** The inclination angle of the airfoil to the freestream flow.

### Output
- **Camber Line Plot:** Displays the shape of the airfoil's camber line.
- **Computed Values:**
  - Lift coefficient (Cl)
  - Moment coefficient (Cm)
- **Streamline Visualization:** Shows the vector field around the airfoil.

## Example
After inputting the parameters and clicking the 'Compute' button, the app will output:
```
Angle of Attack (α): 3.0°
Lift Coefficient (Cl): 0.523
Moment Coefficient (Cm): -0.025
```
Along with the camber line and streamline plots.

## License
This project is open-source and available under the MIT License.

## Author
Developed by [Your Name]
