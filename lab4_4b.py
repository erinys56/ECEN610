import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.interpolate import interp1d

fs = 10e9
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

# Mismatch parameters
offset = 10e-3
time_skew = 5e-12
tau_A = 30e-12
tau_B = 50e-12

# Channel A
x_A = sum(A * np.sin(2 * np.pi * f * t_even) for A, f in zip(amplitudes, frequencies))
x_A *= np.exp(-Ts / tau_A)

# Channel B
t_odd_skewed = t_odd + time_skew
x_B = sum(A * np.sin(2 * np.pi * f * t_odd_skewed) for A, f in zip(amplitudes, frequencies))
x_B *= np.exp(-Ts / tau_B)
x_B += offset

# Interleaved TI-ADC Output
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


x_A_off = ti_adc_output[even_idx].copy()
x_B_off = ti_adc_output[odd_idx].copy()
x_A_off -= np.mean(x_A_off)
x_B_off -= np.mean(x_B_off)


interp_func = interp1d(t_odd + time_skew, x_B_off, kind='linear', bounds_error=False, fill_value="extrapolate")
x_B_time_aligned = interp_func(t_odd)


alpha = 0.9
x_B_bandwidth_eq = np.copy(x_B_time_aligned)
x_B_bandwidth_eq[1:] = x_B_bandwidth_eq[1:] - alpha * x_B_bandwidth_eq[:-1]


rms_A = np.sqrt(np.mean(x_A_off**2))
rms_B = np.sqrt(np.mean(x_B_bandwidth_eq**2))
x_B_bandwidth_eq *= (rms_A / rms_B)


ti_adc_full_corrected = np.empty(N)
ti_adc_full_corrected[even_idx] = x_A_off
ti_adc_full_corrected[odd_idx] = x_B_bandwidth_eq


sndr_all, sig_all, noise_all = compute_sndr_verbose(ti_adc_full_corrected)


plt.figure(figsize=(10, 3))
plt.plot(t * 1e9, ti_adc_full_corrected)
plt.xlabel("Time (ns)")
plt.ylabel("Amplitude (V)")
plt.title(f"Fully Corrected TI-ADC Output (SNDR ≈ {sndr_all:.2f} dB)")
plt.grid(True)
plt.tight_layout()
plt.show()


print(f"[Fully Corrected] Signal Power: {sig_all:.4f}")
print(f"[Fully Corrected] Noise + Distortion Power: {noise_all:.4f}")
print(f"[Fully Corrected] SNDR: {sndr_all:.2f} dB")
