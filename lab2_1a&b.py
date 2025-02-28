import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # pandas 추가

# 1) 샘플링 설정
fs = 5.0001e6  # 샘플링 주파수 (5 MHz)
f_tone = 2e6  # 톤 신호 주파수 (2 MHz)
A = 1  # 진폭 1V
t = np.arange(0, 400e-3, 1/fs)  # 샘플링 시간 400ms

# 2) 사인파 생성
signal = A * np.sin(2 * np.pi * f_tone * t)

# 3) 목표 SNR = 50 dB에 맞춰 Gaussian Noise 추가
SNR_target = 50  # dB

# 4) 신호 전력 계산
signal_power = np.mean(signal**2)

# 5) 노이즈 전력 및 표준 편차 계산
noise_power = signal_power / (10**(SNR_target / 10))
noise_std = np.sqrt(noise_power)  # 표준 편차

# 6) 가우시안 노이즈 생성 및 추가
noise = np.random.normal(0, noise_std, len(signal))
noisy_signal = signal + noise

# 7) 균등 분포(Uniform Noise) 노이즈 분산 계산
uniform_noise_variance = 3 * noise_power
uniform_noise_max = np.sqrt(3 * noise_power)

# 8) 균등 분포 노이즈 생성 및 추가
uniform_noise = np.random.uniform(-uniform_noise_max, uniform_noise_max, len(signal))
uniform_noisy_signal = signal + uniform_noise

# 9) FFT 수행 (1a: 윈도우 없음)
N = len(noisy_signal)
frequencies = np.fft.fftfreq(N, d=1/fs)

fft_noisy_no_window = np.fft.fft(noisy_signal)
psd_noisy_no_window = np.abs(fft_noisy_no_window)**2 / N

fft_uniform_no_window = np.fft.fft(uniform_noisy_signal)
psd_uniform_no_window = np.abs(fft_uniform_no_window)**2 / N

# 10) 신호 피크 및 노이즈 플로어 계산 (1a)
signal_peak_index = np.argmax(psd_noisy_no_window[:N//2])
signal_power_dft_no_window = np.mean(psd_noisy_no_window[max(0, signal_peak_index-2):signal_peak_index+3])
noise_floor_no_window = np.mean(psd_noisy_no_window[N//4:N//2])

signal_power_dft_uniform_no_window = np.mean(psd_uniform_no_window[max(0, signal_peak_index-2):signal_peak_index+3])
noise_floor_uniform_no_window = np.mean(psd_uniform_no_window[N//4:N//2])

# 11) 1a: SNR 계산 (윈도우 없이)
computed_SNR_no_window_gaussian = 10 * np.log10(np.maximum(signal_power_dft_no_window / noise_floor_no_window, 1e-12))
computed_SNR_no_window_uniform = 10 * np.log10(np.maximum(signal_power_dft_uniform_no_window / noise_floor_uniform_no_window, 1e-12))

# 12) 1b: 창 함수 적용 및 FFT 수행
windows = {
    "Hanning": np.hanning(N),
    "Hamming": np.hamming(N),
    "Blackman": np.blackman(N)
}

# 창 함수별 전력 보정 계수 (Power Compensation Factor)
window_compensation = {
    "Hanning": 1.5,
    "Hamming": 1.36,
    "Blackman": 1.67
}

computed_SNR_results = {
    "No Window": {"Gaussian": computed_SNR_no_window_gaussian, "Uniform": computed_SNR_no_window_uniform, "Gaussian Variance": noise_power, "Uniform Variance": uniform_noise_variance}
}

for win_name, window in windows.items():
    # FFT 수행 (Gaussian Noise)
    fft_noisy = np.fft.fft(noisy_signal * window)
    psd_noisy = np.abs(fft_noisy)**2 / N

    # FFT 수행 (Uniform Noise)
    fft_uniform = np.fft.fft(uniform_noisy_signal * window)
    psd_uniform = np.abs(fft_uniform)**2 / N

    # 신호 피크 및 노이즈 플로어 계산
    signal_peak_index = np.argmax(psd_noisy[:N//2])
    signal_power_dft = np.mean(psd_noisy[max(0, signal_peak_index-2):signal_peak_index+3])
    noise_floor = np.mean(psd_noisy[N//4:N//2])

    signal_power_dft_uniform = np.mean(psd_uniform[max(0, signal_peak_index-2):signal_peak_index+3])
    noise_floor_uniform = np.mean(psd_uniform[N//4:N//2])

    # 창 함수 보정 계수 적용 (신호 전력 보정)
    signal_power_dft /= window_compensation[win_name]
    signal_power_dft_uniform /= window_compensation[win_name]

    # SNR 계산 (Gaussian Noise)
    computed_SNR_g = 10 * np.log10(np.maximum(signal_power_dft / noise_floor, 1e-12))

    # SNR 계산 (Uniform Noise)
    computed_SNR_u = 10 * np.log10(np.maximum(signal_power_dft_uniform / noise_floor_uniform, 1e-12))

    computed_SNR_results[win_name] = {"Gaussian": computed_SNR_g, "Uniform": computed_SNR_u, "Gaussian Variance": noise_power, "Uniform Variance": uniform_noise_variance}

# 결과 출력 (테이블 형태)
df_results = pd.DataFrame(computed_SNR_results).T
print(df_results)

# 그래프 출력
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
