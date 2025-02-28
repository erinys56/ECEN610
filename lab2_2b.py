import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

def quantizer(signal, bits):
    levels = 2 ** bits
    delta = 2 / levels
    return np.round(signal / delta) * delta

def compute_psd_and_snr(original_signal, quantized_signal, fs):
    frequencies, psd = welch(quantized_signal, fs, nperseg=min(4096, len(quantized_signal)))
    psd_db = 10 * np.log10(psd + 1e-12)
    signal_power = np.var(original_signal)
    noise_power = np.var(original_signal - quantized_signal)
    snr = 10 * np.log10(signal_power / noise_power)
    return float(snr), frequencies, psd_db

fs = 5 * 400.1e6
fin = 200e6
num_periods_30, num_periods_100 = 30, 100

t_30 = np.arange(0, num_periods_30 / fin, 1 / fs)
t_100 = np.arange(0, num_periods_100 / fin, 1 / fs)

input_signal_30 = np.sin(2 * np.pi * fin * t_30)
input_signal_100 = np.sin(2 * np.pi * fin * t_100)

quantized_signal_30_6bit = quantizer(input_signal_30, 6)
quantized_signal_100_6bit = quantizer(input_signal_100, 6)

snr_30_6bit, freq_30, psd_30 = compute_psd_and_snr(input_signal_30, quantized_signal_30_6bit, fs)
snr_100_6bit, freq_100, psd_100 = compute_psd_and_snr(input_signal_100, quantized_signal_100_6bit, fs)

print("B) 6-bit Quantization with Higher Sampling Rate:")
print(f"6-bit SNR (30 periods) with higher fs: {snr_30_6bit:.2f} dB")
print(f"6-bit SNR (100 periods) with higher fs: {snr_100_6bit:.2f} dB")

def plot_psd(freq, psd, title):
    plt.figure()
    plt.plot(freq, psd)
    plt.xlim(0, fs / 2)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density (dB/Hz)")
    plt.title(title)
    plt.grid()
    plt.show()

plot_psd(freq_30, psd_30, "PSD of 6-bit Quantized Signal (30 periods, Higher fs)")
plot_psd(freq_100, psd_100, "PSD of 6-bit Quantized Signal (100 periods, Higher fs)")
