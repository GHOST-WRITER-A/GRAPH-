import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("ad7746_binary_converted (1).csv")

# Extract columns
t = df["Time_s"]
c = df["Differential_capacitance_pF"]

# Plot
plt.figure(figsize=(10, 5))
plt.plot(t, c, linewidth=0.8, color="tab:blue")
plt.xlabel("Time (s)")
plt.ylabel("Differential Capacitance (pF)")
plt.title("Capacitance vs Time (AD7746)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("c_vs_t.png", dpi=200)
plt.show()
