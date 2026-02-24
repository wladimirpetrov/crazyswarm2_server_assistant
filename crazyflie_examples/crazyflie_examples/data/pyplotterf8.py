import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv('figure8.csv')

# Evaluate a trajectory for the first row
row = df.iloc[0]
duration = row['duration']
t = np.linspace(0, duration, 100)

# Get polynomial coefficients
x_coeffs = [row[f'x^{i}'] for i in range(8)]
y_coeffs = [row[f'y^{i}'] for i in range(8)]
z_coeffs = [row[f'z^{i}'] for i in range(8)]  # Optional: if z isn't all zero

# Evaluate polynomial
x_vals = sum(c * t**i for i, c in enumerate(x_coeffs))
y_vals = sum(c * t**i for i, c in enumerate(y_coeffs))
z_vals = sum(c * t**i for i, c in enumerate(z_coeffs))  # Optional

# 2D Plot (x-y)
plt.plot(x_vals, y_vals, label='Trajectory')
plt.xlabel('x')
plt.ylabel('y')
plt.title('2D Trajectory')
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.show()
