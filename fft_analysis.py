import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# FFT Analysis of Pressure vs Time Data
# =====================================================================

# ---------- Load data ----------
df = pd.read_csv("pressure_vs_time_calibrated.csv")
t = df["Time_s"].values
p = df["Differential_pressure_Pa"].values

n = len(p)
dt = t[1] - t[0]           # sample spacing (s)
fs = 1 / dt                # sample rate (Hz)
record_length = n * dt     # total duration (s)
freq_resolution = 1 / record_length   # Hz per FFT bin
nyquist = fs / 2           # highest resolvable frequency (Hz)

print(f"N samples        : {n}")
print(f"Sample rate (fs)  : {fs:.3f} Hz")
print(f"Record length     : {record_length:.3f} s")
print(f"Frequency resolution (Δf) : {freq_resolution:.4f} Hz")
print(f"Nyquist frequency : {nyquist:.3f} Hz")
print()

# ---------- Preprocess ----------
# 1) Remove DC offset (mean) - otherwise it dominates the spectrum
p_detrend = p - p.mean()

# 2) Apply a Hann window to reduce spectral leakage
window = np.hanning(n)
p_windowed = p_detrend * window

# Window correction factor (so amplitude values stay meaningful)
window_correction = 1 / np.mean(window)

# ---------- FFT ----------
freqs = np.fft.rfftfreq(n, d=dt)
fft_vals = np.fft.rfft(p_windowed)

# Amplitude spectrum (normalized so it reads in real Pa units)
amplitude = (2 / n) * np.abs(fft_vals) * window_correction
amplitude[0] = amplitude[0] / 2  # DC bin has no factor-of-2

# ---------- Find peaks ----------
# Skip the DC bin (index 0) when searching for peaks
peak_idx = np.argsort(amplitude[1:])[::-1][:10] + 1

print("Top 10 frequency components:")
print(f"{'Freq (Hz)':>10} {'Period (s)':>12} {'Amplitude (Pa)':>16}")
for i in peak_idx:
    period = 1 / freqs[i] if freqs[i] > 0 else float('inf')
    print(f"{freqs[i]:10.4f} {period:12.3f} {amplitude[i]:16.4f}")

dominant_i = peak_idx[0]
print(f"\nDominant frequency: {freqs[dominant_i]:.4f} Hz "
      f"(period {1/freqs[dominant_i]:.3f} s), amplitude {amplitude[dominant_i]:.4f} Pa")

# ---------- Plots ----------
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Time domain
axes[0].plot(t, p, linewidth=0.8, color="tab:red")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Pressure (Pa)")
axes[0].set_title("Time Domain: Pressure vs Time")
axes[0].grid(True, alpha=0.3)

# Frequency domain
axes[1].plot(freqs, amplitude, color="tab:purple")
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Amplitude (Pa)")
axes[1].set_title("Frequency Domain: FFT Spectrum (Hann windowed)")
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(0, nyquist)

plt.tight_layout()
plt.savefig("fft_analysis.png", dpi=200)
plt.show()

print("\nSaved plot: fft_analysis.png")
