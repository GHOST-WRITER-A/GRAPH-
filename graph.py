import matplotlib.pyplot as plt
import numpy as np

# Experimental data
pressure_pa = np.array([
    0.00, 19.62, 39.24, 58.86, 78.48,
    98.10, 117.72, 137.34, 156.96, 176.58, 196.20
])

capacitance_pf = np.array([
    0.20, 0.27, 0.65, 1.66, 1.73,
    2.40, 2.80, 2.82, 3.59, 3.716, 4.096
])

# Linear regression
slope, intercept = np.polyfit(pressure_pa, capacitance_pf, 1)

# Calculate fitted values
fit = slope * pressure_pa + intercept

# Calculate R²
r2 = 1 - np.sum((capacitance_pf - fit)**2) / \
         np.sum((capacitance_pf - np.mean(capacitance_pf))**2)

# Plot experimental data
plt.figure(figsize=(9, 5.5))

plt.plot(
    pressure_pa,
    capacitance_pf,
    marker='o',
    linewidth=2,
    markersize=6,
    label='Experimental Data'
)

# Plot linear trendline
plt.plot(
    pressure_pa,
    fit,
    linestyle='--',
    linewidth=1.8,
    label=f'Linear Fit: C = {slope:.4f}ΔP + {intercept:.3f}'
)

# Labels and title
plt.xlabel('Differential Pressure, ΔP (Pa)', fontsize=12)
plt.ylabel('Capacitance (pF)', fontsize=12)
plt.title('Differential Pressure vs Capacitance', fontsize=14)

# Grid
plt.grid(True, alpha=0.3)

# Legend
plt.legend()

# Make layout clean
plt.tight_layout()

# Show plot
plt.show()

# Print results
print(f"Linear equation:")
print(f"C = {slope:.5f} ΔP + {intercept:.5f}")
print(f"Sensitivity = {slope:.5f} pF/Pa")
print(f"R² = {r2:.5f}")