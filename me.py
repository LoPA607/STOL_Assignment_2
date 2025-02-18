import numpy as np
import matplotlib.pyplot as plt

def generate_camber_line(m, p, num_points=200):
    """
    Generate the camber line for a NACA 4-digit airfoil.
    
    Parameters:
        m (float): Maximum camber (fraction of chord length).
        p (float): Location of maximum camber (fraction of chord length).
        num_points (int): Number of points along the chord to calculate.
    
    Returns:
        x (numpy array): Chordwise coordinates (x).
        yc (numpy array): Camber line coordinates (y).
    """
    # Generate x-coordinates along the chord (from 0 to 1)
    x = np.linspace(0, 1, num_points)
    
    # Initialize y-coordinates for the camber line
    yc = np.zeros_like(x)
    
    # Compute y-coordinates for the forward section (0 <= x <= p)
    forward_mask = x <= p
    yc[forward_mask] = (m / p**2) * (2 * p * x[forward_mask] - x[forward_mask]**2)
    
    # Compute y-coordinates for the rear section (p < x <= 1)
    rear_mask = x > p
    yc[rear_mask] = (m / (1 - p)**2) * ((1 - 2 * p) + 2 * p * x[rear_mask] - x[rear_mask]**2)
    
    return x, yc


def compute_slope(x, yc):
    """
    Compute the slope of the camber line using finite differences.
    
    Parameters:
        x (numpy array): Chordwise coordinates (x).
        yc (numpy array): Camber line coordinates (y).
    
    Returns:
        slope (numpy array): Slope of the camber line (dy/dx).
    """
    # Compute the slope using central finite differences
    slope = np.gradient(yc, x)
    return slope


def plot_camber_and_slope(x, yc, slope):
    """
    Plot the camber line and its slope.
    
    Parameters:
        x (numpy array): Chordwise coordinates (x).
        yc (numpy array): Camber line coordinates (y).
        slope (numpy array): Slope of the camber line (dy/dx).
    """
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    
    # Plot the camber line
    ax1.plot(x, yc, label="Camber Line", color="blue")
    ax1.set_title("Camber Line")
    ax1.set_xlabel("Chordwise Coordinate (x)")
    ax1.set_ylabel("Camber Line (y)")
    ax1.grid(True)
    ax1.legend()
    
    # Plot the slope of the camber line
    ax2.plot(x, slope, label="Slope (dy/dx)", color="red")
    ax2.set_title("Slope of Camber Line")
    ax2.set_xlabel("Chordwise Coordinate (x)")
    ax2.set_ylabel("Slope (dy/dx)")
    ax2.grid(True)
    ax2.legend()
    
    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


# Main script
if __name__ == "__main__":
    # Define parameters for the NACA airfoil
    m = 0.02  # Maximum camber (2% of chord)
    p = 0.4   # Location of maximum camber (40% of chord)
    
    # Generate the camber line
    x, yc = generate_camber_line(m, p)
    
    # Compute the slope of the camber line
    slope = compute_slope(x, yc)
    
    # Plot the camber line and its slope
    plot_camber_and_slope(x, yc, slope)