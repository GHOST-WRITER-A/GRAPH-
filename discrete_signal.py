import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# Discrete-Time Signal Plot: Pressure vs Sample Index / Time
# =====================================================================

# ---------- Load data ----------
df = pd.read_csv("pressure_vs_time_calibrated.csv")
t = df["Time_s"].values
p = df["Differential_pressure_Pa"].values

n = len(p)
dt = t[1] - t[0]
fs = 1 / dt

print(f"N samples : {n}")
print(f"Sample rate (fs) : {fs:.3f} Hz")
print(f"Sample spacing (dt) : {dt} s")

# ---------- Discrete sample index n (as used in DSP notation x[n]) ----------
sample_index = np.arange(n)

# =====================================================================
# For a full 1000-point record, a stem plot is unreadable (way too dense).
# So: 1) show a ZOOMED-IN discrete stem plot for the first ~50 samples,
#     2) show the FULL record as a discrete signal using a fast marker-only
#        style (still discrete samples, just without individual stems).
# =====================================================================

fig, axes = plt.subplots(2, 1, figsize=(11, 9))

# --- Plot 1: True stem plot, zoomed to first 50 samples ---
n_zoom = 50
axes[0].stem(sample_index[:n_zoom], p[:n_zoom], basefmt=" ")
axes[0].set_xlabel("Sample index n")
axes[0].set_ylabel("Pressure p[n] (Pa)")
axes[0].set_title(f"Discrete-Time Signal p[n] — first {n_zoom} samples "
                   f"(fs = {fs:.0f} Hz)")
axes[0].grid(True, alpha=0.3)

# --- Plot 2: Full record shown as discrete samples (markers, no stems) ---
axes[1].plot(t, p, linestyle="none", marker=".", markersize=3, color="tab:red")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Pressure p[n] (Pa)")
axes[1].set_title(f"Full Discrete-Time Signal — all {n} samples")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("discrete_signal.png", dpi=200)
plt.show()

print("\nSaved plot: discrete_signal.png")
print("Top plot  : true stem plot (x[n] vs n), zoomed to first 50 samples")
print("Bottom plot: full 1000-sample record, discrete markers vs time")
