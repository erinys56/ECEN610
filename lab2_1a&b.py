import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

fs = 5.0001e6
f_tone = 2e6
A = 1
t = np.arange(0, 400e-3, 1/fs)

signal = A * np.sin(2 * np.pi * f_tone * t)

SNR_target = 50

signal_power = np.mean(signal**2)

noise_power = signal_power / (10**(SNR_target / 10))
noise_std = np.sqrt(noise_power)

noise = np.random.normal(0, noise_std, len(signal))
noisy_signal = signal + noise

uniform_noise_variance = 3 * noise_power
uniform_noise_max = np.sqrt(3 * noise_power)

uniform_noise = np.random.uniform(-uniform_noise_max, uniform_noise_max, len(signal))
uniform_noisy_signal = signal + uniform_noise

N = len(noisy_signal)
frequencies = np.fft.fftfreq(N, d=1/fs)

fft_noisy_no_window = np.fft.fft(noisy_signal)
psd_noisy_no_window = np.abs(fft_noisy_no_window)**2 / N

fft_uniform_no_window = np.fft.fft(uniform_noisy_signal)
psd_uniform_no_window = np.abs(fft_uniform_no_window)**2 / N

signal_peak_index = np.argmax(psd_noisy_no_window[:N//2])
signal_power_dft_no_window = np.mean(psd_noisy_no_window[max(0, signal_peak_index-2):signal_peak_index+3])
noise_floor_no_window = np.mean(psd_noisy_no_window[N//4:N//2])

signal_power_dft_uniform_no_window = np.mean(psd_uniform_no_window[max(0, signal_peak_index-2):signal_peak_index+3])
noise_floor_uniform_no_window = np.mean(psd_uniform_no_window[N//4:N//2])

computed_SNR_no_window_gaussian = 10 * np.log10(np.maximum(signal_power_dft_no_window / noise_floor_no_window, 1e-12))
computed_SNR_no_window_uniform = 10 * np.log10(np.maximum(signal_power_dft_uniform_no_window / noise_floor_uniform_no_window, 1e-12))

windows = {
    "Hanning": np.hanning(N),
    "Hamming": np.hamming(N),
    "Blackman": np.blackman(N)
}

window_compensation = {
    "Hanning": 1.5,
    "Hamming": 1.36,
    "Blackman": 1.67
}

computed_SNR_results = {
    "No Window": {"Gaussian": computed_SNR_no_window_gaussian, "Uniform": computed_SNR_no_window_uniform, "Gaussian Variance": noise_power, "Uniform Variance": uniform_noise_variance}
}

for win_name, window in windows.items():
    fft_noisy = np.fft.fft(noisy_signal * window)
    psd_noisy = np.abs(fft_noisy)**2 / N

    fft_uniform = np.fft.fft(uniform_noisy_signal * window)
    psd_uniform = np.abs(fft_uniform)**2 / N

    signal_peak_index = np.argmax(psd_noisy[:N//2])
    signal_power_dft = np.mean(psd_noisy[max(0, signal_peak_index-2):signal_peak_index+3])
    noise_floor = np.mean(psd_noisy[N//4:N//2])

    signal_power_dft_uniform = np.mean(psd_uniform[max(0, signal_peak_index-2):signal_peak_index+3])
    noise_floor_uniform = np.mean(psd_uniform[N//4:N//2])

    signal_power_dft /= window_compensation[win_name]
    signal_power_dft_uniform /= window_compensation[win_name]

    computed_SNR_g = 10 * np.log10(np.maximum(signal_power_dft / noise_floor, 1e-12))
    computed_SNR_u = 10 * np.log10(np.maximum(signal_power_dft_uniform / noise_floor_uniform, 1e-12))

    computed_SNR_results[win_name] = {"Gaussian": computed_SNR_g, "Uniform": computed_SNR_u, "Gaussian Variance": noise_power, "Uniform Variance": uniform_noise_variance}

df_results = pd.DataFrame(computed_SNR_results).T
print(df_results)

plt.figure(figsize=(8, 4))
bar_width = 0.3
positions = np.arange(len(computed_SNR_results))

plt.bar(positions - bar_width/2, [computed_SNR_results[w]["Gaussian"] for w in computed_SNR_results.keys()], 
        bar_width, label="Gaussian Noise")
plt.bar(positions + bar_width/2, [computed_SNR_results[w]["Uniform"] for w in computed_SNR_results.keys()], 
        bar_width, label="Uniform Noise")

plt.xticks(positions, computed_SNR_results.keys())
plt.xlabel("Window Function")
plt.ylabel("Computed SNR (dB)")
plt.title("SNR with and without Window Functions")
plt.legend()
plt.grid()
plt.show()
