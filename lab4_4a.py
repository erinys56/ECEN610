import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

# ----- Parameters -----
fs = 10e9                # Sampling frequency
Ts = 1 / fs
N = 2048
t = np.arange(N) * Ts


frequencies = [0.2e9, 0.58e9, 1.0e9, 1.7e9, 2.4e9]
amplitudes = [0.1] * len(frequencies)
x = sum(A * np.sin(2 * np.pi * f * t) for A, f in zip(amplitudes, frequencies))


even_idx = np.arange(0, N, 2)
odd_idx = np.arange(1, N, 2)
t_even = t[even_idx]
t_odd = t[odd_idx]


offset = 10e-3              # 10 mV offset
time_skew = 5e-12           # 5 ps time mismatch
tau_A = 30e-12              # RC time constant (channel A)
tau_B = 50e-12              # RC time constant (channel B)


x_A = sum(A * np.sin(2 * np.pi * f * t_even) for A, f in zip(amplitudes, frequencies))
x_A *= np.exp(-Ts / tau_A)


t_odd_skewed = t_odd + time_skew
x_B = sum(A * np.sin(2 * np.pi * f * t_odd_skewed) for A, f in zip(amplitudes, frequencies))
x_B *= np.exp(-Ts / tau_B)
x_B += offset


ti_adc_output = np.empty(N)
ti_adc_output[even_idx] = x_A
ti_adc_output[odd_idx] = x_B


def compute_sndr_verbose(signal):
    windowed = signal * np.hanning(len(signal))
    Y = fft(windowed)
    Y_mag = np.abs(Y[:len(Y)//2])**2
    signal_power = np.max(Y_mag)
    noise_power = np.sum(Y_mag) - signal_power
    sndr = 10 * np.log10(signal_power / noise_power)
    return sndr, signal_power, noise_power

sndr, sig_power, noise_power = compute_sndr_verbose(ti_adc_output)


print(f"Signal Power: {sig_power:.4f}")
print(f"Noise + Distortion Power: {noise_power:.4f}")
print(f"SNDR: {sndr:.2f} dB")


plt.figure(figsize=(10, 3))
plt.plot(t * 1e9, ti_adc_output)
plt.xlabel("Time (ns)")
plt.ylabel("Amplitude (V)")
plt.title(f"TI-ADC Output with Mismatches (SNDR ≈ {sndr:.2f} dB)")
plt.grid(True)
plt.tight_layout()
plt.show()
