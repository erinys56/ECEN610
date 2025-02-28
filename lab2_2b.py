import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# 양자화기 구현
def quantizer(signal, bits):
    levels = 2 ** bits  # 양자화 레벨 수
    delta = 2 / levels  # 양자화 스텝 크기 (-1 ~ 1 범위)
    return np.round(signal / delta) * delta  # 양자화 수행

# PSD 및 SNR 계산 함수
def compute_psd_and_snr(original_signal, quantized_signal, fs):
    frequencies, psd = welch(quantized_signal, fs, nperseg=min(4096, len(quantized_signal)))
    psd_db = 10 * np.log10(psd + 1e-12)  # dB 변환
    
    # SNR 계산
    signal_power = np.var(original_signal)
    noise_power = np.var(original_signal - quantized_signal)
    snr = 10 * np.log10(signal_power / noise_power)
    
    return float(snr), frequencies, psd_db  # ✅ SNR을 float로 변환하여 반환

# 샘플링 및 신호 설정
fs = 5 * 400.1e6  # ✅ Nyquist Rate보다 5배 큰 샘플링 주파수로 설정
fin = 200e6  # 입력 신호 주파수 (200 MHz)
num_periods_30, num_periods_100 = 30, 100

# 시간 벡터 생성
t_30 = np.arange(0, num_periods_30 / fin, 1 / fs)
t_100 = np.arange(0, num_periods_100 / fin, 1 / fs)

# 사인파 생성 (-1 ~ 1 풀 스케일)
input_signal_30 = np.sin(2 * np.pi * fin * t_30)
input_signal_100 = np.sin(2 * np.pi * fin * t_100)

# 6비트 양자화
quantized_signal_30_6bit = quantizer(input_signal_30, 6)
quantized_signal_100_6bit = quantizer(input_signal_100, 6)

# SNR 및 PSD 계산
snr_30_6bit, freq_30, psd_30 = compute_psd_and_snr(input_signal_30, quantized_signal_30_6bit, fs)
snr_100_6bit, freq_100, psd_100 = compute_psd_and_snr(input_signal_100, quantized_signal_100_6bit, fs)

# 결과 출력
print("B) 6-bit Quantization with Higher Sampling Rate:")
print(f"6-bit SNR (30 periods) with higher fs: {snr_30_6bit:.2f} dB")
print(f"6-bit SNR (100 periods) with higher fs: {snr_100_6bit:.2f} dB")

# PSD 플롯
def plot_psd(freq, psd, title):
    plt.figure()
    plt.plot(freq, psd)  # 로그 스케일 대신 일반 스케일 확인
    plt.xlim(0, fs / 2)  # Nyquist 주파수까지만 표시
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density (dB/Hz)")
    plt.title(title)
    plt.grid()
    plt.show()

plot_psd(freq_30, psd_30, "PSD of 6-bit Quantized Signal (30 periods, Higher fs)")
plot_psd(freq_100, psd_100, "PSD of 6-bit Quantized Signal (100 periods, Higher fs)")