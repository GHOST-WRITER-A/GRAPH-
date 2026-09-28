import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ---------- Load data ----------
df = pd.read_csv("pressure_vs_time_calibrated.csv")
t = df["Time_s"].values
p = df["Differential_pressure_Pa"].values

# ---------- Plot P vs t ----------
plt.figure(figsize=(10, 5))
plt.plot(t, p, linewidth=0.8, color="tab:red")
plt.xlabel("Time (s)")
plt.ylabel("Differential Pressure (Pa)")
plt.title("Pressure vs Time")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("p_vs_t.png", dpi=200)
plt.show()

# ---------- R^2 helper ----------
def r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot

print(f"N points: {len(t)}")
print(f"P range: {p.min():.3f} to {p.max():.3f} Pa")
print(f"P mean: {p.mean():.3f} Pa, std: {p.std():.3f} Pa\n")

# ---------- Linear fit ----------
p1 = np.polyfit(t, p, 1)
print(f"Linear fit:    P = {p1[0]:.5f}*t + {p1[1]:.3f}   R^2 = {r2(p, np.polyval(p1, t)):.4f}")

# ---------- Quadratic fit ----------
p2 = np.polyfit(t, p, 2)
print(f"Quadratic fit: R^2 = {r2(p, np.polyval(p2, t)):.4f}")

# ---------- FFT (frequency content) ----------
dt = t[1] - t[0]
fs = 1 / dt
n = len(p)
p_detrend = p - p.mean()

freqs = np.fft.rfftfreq(n, d=dt)
fft_vals = np.fft.rfft(p_detrend)
mag = np.abs(fft_vals)

# Top 8 frequency peaks (skip DC bin)
idx = np.argsort(mag[1:])[::-1][:8] + 1
print("\nTop FFT peaks:")
for i in idx:
    print(f"  f = {freqs[i]:.3f} Hz   mag = {mag[i]:.1f}   period = {1/freqs[i]:.3f} s")

# Plot the spectrum too
plt.figure(figsize=(10, 4))
plt.plot(freqs, mag, color="tab:purple")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Frequency Spectrum of Pressure Signal")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("p_spectrum.png", dpi=200)
plt.show()

# ---------- Sinusoidal fit ----------
def sine_func(tt, A, f, phi, offset):
    return A * np.sin(2 * np.pi * f * tt + phi) + offset

print("\nSinusoidal fits (different starting-frequency guesses):")
best = None
for f0 in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5]:
    try:
        popt, _ = curve_fit(sine_func, t, p, p0=[p.std(), f0, 0, p.mean()], maxfev=5000)
        yhat = sine_func(t, *popt)
        score = r2(p, yhat)
        print(f"  guess f0={f0:.2f} Hz -> fit f={popt[1]:.4f} Hz, A={popt[0]:.3f}, R^2={score:.4f}")
        if best is None or score > best[0]:
            best = (score, popt)
    except RuntimeError:
        print(f"  guess f0={f0:.2f} Hz -> fit failed to converge")

if best:
    print(f"\nBest single-sinusoid fit: R^2 = {best[0]:.4f}, params (A, f, phi, offset) = {best[1]}")

print("\nVerdict: R^2 well below ~0.9 for every model means P(t) is NOT")
print("well described by a simple linear, quadratic, or single-sinusoid")
print("relation - the signal is dominated by broadband noise/drift, not")
print("a clean deterministic pattern.")