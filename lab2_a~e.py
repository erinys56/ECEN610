import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, windows

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
    return float(snr)

def compute_snr_with_hanning(original_signal, quantized_signal, fs):
    window = windows.hann(len(original_signal))
    windowed_original = original_signal * window
    windowed_quantized = quantized_signal * window
    return compute_psd_and_snr(windowed_original, windowed_quantized, fs)

fs = 400.1e6
fin = 200e6
num_periods_30, num_periods_100 = 30, 100

t_30 = np.arange(0, num_periods_30 / fin, 1 / fs)
t_100 = np.arange(0, num_periods_100 / fin, 1 / fs)

input_signal_30 = np.sin(2 * np.pi * fin * t_30)
input_signal_100 = np.sin(2 * np.pi * fin * t_100)

quantized_signal_30_6bit = quantizer(input_signal_30, 6)
quantized_signal_100_6bit = quantizer(input_signal_100, 6)
quantized_signal_30_12bit = quantizer(input_signal_30, 12)
quantized_signal_100_12bit = quantizer(input_signal_100, 12)

snr_30_6bit = compute_psd_and_snr(input_signal_30, quantized_signal_30_6bit, fs)
snr_100_6bit = compute_psd_and_snr(input_signal_100, quantized_signal_100_6bit, fs)
snr_30_12bit = compute_psd_and_snr(input_signal_30, quantized_signal_30_12bit, fs)
snr_100_12bit = compute_psd_and_snr(input_signal_100, quantized_signal_100_12bit, fs)

snr_30_6bit_hanning = compute_snr_with_hanning(input_signal_30, quantized_signal_30_6bit, fs)
snr_100_6bit_hanning = compute_snr_with_hanning(input_signal_100, quantized_signal_100_6bit, fs)
snr_30_12bit_hanning = compute_snr_with_hanning(input_signal_30, quantized_signal_30_12bit, fs)
snr_100_12bit_hanning = compute_snr_with_hanning(input_signal_100, quantized_signal_100_12bit, fs)

target_snr_db = 38
signal_power = np.var(input_signal_30)
noise_power_target = signal_power / (10 ** (target_snr_db / 10))
noise_std_target = np.sqrt(noise_power_target)
np.random.seed(42)
input_signal_noisy_30 = input_signal_30 + noise_std_target * np.random.randn(len(input_signal_30))
input_signal_noisy_100 = input_signal_100 + noise_std_target * np.random.randn(len(input_signal_100))

quantized_signal_noisy_30_6bit = quantizer(input_signal_noisy_30, 6)
quantized_signal_noisy_100_6bit = quantizer(input_signal_noisy_100, 6)
quantized_signal_noisy_30_12bit = quantizer(input_signal_noisy_30, 12)
quantized_signal_noisy_100_12bit = quantizer(input_signal_noisy_100, 12)

snr_noisy_30_6bit = compute_psd_and_snr(input_signal_noisy_30, quantized_signal_noisy_30_6bit, fs)
snr_noisy_100_6bit = compute_psd_and_snr(input_signal_noisy_100, quantized_signal_noisy_100_6bit, fs)
snr_noisy_30_12bit = compute_psd_and_snr(input_signal_noisy_30, quantized_signal_noisy_30_12bit, fs)
snr_noisy_100_12bit = compute_psd_and_snr(input_signal_noisy_100, quantized_signal_noisy_100_12bit, fs)

snr_noisy_30_6bit_hanning = compute_snr_with_hanning(input_signal_noisy_30, quantized_signal_noisy_30_6bit, fs)
snr_noisy_100_6bit_hanning = compute_snr_with_hanning(input_signal_noisy_100, quantized_signal_noisy_100_6bit, fs)
snr_noisy_30_12bit_hanning = compute_snr_with_hanning(input_signal_noisy_30, quantized_signal_noisy_30_12bit, fs)
snr_noisy_100_12bit_hanning = compute_snr_with_hanning(input_signal_noisy_100, quantized_signal_noisy_100_12bit, fs)

print(f"6-bit SNR (30 periods) before noise: {snr_30_6bit:.2f} dB, after noise: {snr_noisy_30_6bit:.2f} dB")
print(f"6-bit SNR (100 periods) before noise: {snr_100_6bit:.2f} dB, after noise: {snr_noisy_100_6bit:.2f} dB")
print(f"12-bit SNR (30 periods) before noise: {snr_30_12bit:.2f} dB, after noise: {snr_noisy_30_12bit:.2f} dB")
print(f"12-bit SNR (100 periods) before noise: {snr_100_12bit:.2f} dB, after noise: {snr_noisy_100_12bit:.2f} dB")
print(f"6-bit SNR (30 periods) after Hanning: {snr_30_6bit_hanning:.2f} dB, after noise + Hanning: {snr_noisy_30_6bit_hanning:.2f} dB")
print(f"6-bit SNR (100 periods) after Hanning: {snr_100_6bit_hanning:.2f} dB, after noise + Hanning: {snr_noisy_100_6bit_hanning:.2f} dB")
print(f"12-bit SNR (30 periods) after Hanning: {snr_30_12bit_hanning:.2f} dB, after noise + Hanning: {snr_noisy_30_12bit_hanning:.2f} dB")
print(f"12-bit SNR (100 periods) after Hanning: {snr_100_12bit_hanning:.2f} dB, after noise + Hanning: {snr_noisy_100_12bit_hanning:.2f} dB")
