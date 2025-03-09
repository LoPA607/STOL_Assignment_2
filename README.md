# Thin Airfoil Analysis

This project focuses on airfoil analysis using thin airfoil theory. There are two versions available:

1. **NACA Airfoil Analysis**:
   - Uses `GUI_naca.md` for the graphical user interface.
   - Executes with `final.py`.

2. **User-Defined Airfoil Analysis**:
   - Uses `GUI_user_defined.md` for the graphical user interface.
   - Executes with `final2.py`.

## Features
- Define a **custom camber line** as a function of `x`.
- Set **angle of attack (alpha)** in degrees.
- Adjust **freestream velocity**.
- Plot **camber slope** for the given custom camber line.
- Generate and visualize **streamlines (vector field) around the airfoil**.

## Installation
To run the application, ensure you have Python installed along with the required dependencies:

```sh
pip install streamlit numpy matplotlib
```

## Usage
Run the application with:

For NACA airfoil analysis:
```sh
streamlit run final.py
```

For user-defined airfoil analysis:
```sh
streamlit run final2.py
