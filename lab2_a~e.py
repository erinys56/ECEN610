import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, windows

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
    
    return float(snr)  # ✅ SNR을 float로 변환하여 반환

# Hanning 윈도우 적용 후 SNR 계산 함수
def compute_snr_with_hanning(original_signal, quantized_signal, fs):
    window = windows.hann(len(original_signal))
    windowed_original = original_signal * window
    windowed_quantized = quantized_signal * window
    
    return compute_psd_and_snr(windowed_original, windowed_quantized, fs)

# 샘플링 및 신호 설정
fs = 400.1e6  # 비정수 샘플링 (400.1 MHz)
fin = 200e6  # 입력 신호 주파수 (200 MHz)
num_periods_30, num_periods_100 = 30, 100

# 시간 벡터 생성
t_30 = np.arange(0, num_periods_30 / fin, 1 / fs)
t_100 = np.arange(0, num_periods_100 / fin, 1 / fs)

# 사인파 생성 (-1 ~ 1 풀 스케일)
input_signal_30 = np.sin(2 * np.pi * fin * t_30)
input_signal_100 = np.sin(2 * np.pi * fin * t_100)

# 6비트 및 12비트 양자화
quantized_signal_30_6bit = quantizer(input_signal_30, 6)
quantized_signal_100_6bit = quantizer(input_signal_100, 6)
quantized_signal_30_12bit = quantizer(input_signal_30, 12)
quantized_signal_100_12bit = quantizer(input_signal_100, 12)

# A) ~ D) 실험 결과 계산
snr_30_6bit = compute_psd_and_snr(input_signal_30, quantized_signal_30_6bit, fs)
snr_100_6bit = compute_psd_and_snr(input_signal_100, quantized_signal_100_6bit, fs)
snr_30_12bit = compute_psd_and_snr(input_signal_30, quantized_signal_30_12bit, fs)
snr_100_12bit = compute_psd_and_snr(input_signal_100, quantized_signal_100_12bit, fs)

snr_30_6bit_hanning = compute_snr_with_hanning(input_signal_30, quantized_signal_30_6bit, fs)
snr_100_6bit_hanning = compute_snr_with_hanning(input_signal_100, quantized_signal_100_6bit, fs)
snr_30_12bit_hanning = compute_snr_with_hanning(input_signal_30, quantized_signal_30_12bit, fs)
snr_100_12bit_hanning = compute_snr_with_hanning(input_signal_100, quantized_signal_100_12bit, fs)

# 노이즈 추가 (SNR = 38 dB)
target_snr_db = 38
signal_power = np.var(input_signal_30)
noise_power_target = signal_power / (10 ** (target_snr_db / 10))
noise_std_target = np.sqrt(noise_power_target)
np.random.seed(42)
input_signal_noisy_30 = input_signal_30 + noise_std_target * np.random.randn(len(input_signal_30))
input_signal_noisy_100 = input_signal_100 + noise_std_target * np.random.randn(len(input_signal_100))

# 노이즈가 추가된 신호 양자화
quantized_signal_noisy_30_6bit = quantizer(input_signal_noisy_30, 6)
quantized_signal_noisy_100_6bit = quantizer(input_signal_noisy_100, 6)
quantized_signal_noisy_30_12bit = quantizer(input_signal_noisy_30, 12)
quantized_signal_noisy_100_12bit = quantizer(input_signal_noisy_100, 12)

# SNR 계산 (노이즈 추가 후)
snr_noisy_30_6bit = compute_psd_and_snr(input_signal_noisy_30, quantized_signal_noisy_30_6bit, fs)
snr_noisy_100_6bit = compute_psd_and_snr(input_signal_noisy_100, quantized_signal_noisy_100_6bit, fs)
snr_noisy_30_12bit = compute_psd_and_snr(input_signal_noisy_30, quantized_signal_noisy_30_12bit, fs)
snr_noisy_100_12bit = compute_psd_and_snr(input_signal_noisy_100, quantized_signal_noisy_100_12bit, fs)

snr_noisy_30_6bit_hanning = compute_snr_with_hanning(input_signal_noisy_30, quantized_signal_noisy_30_6bit, fs)
snr_noisy_100_6bit_hanning = compute_snr_with_hanning(input_signal_noisy_100, quantized_signal_noisy_100_6bit, fs)
snr_noisy_30_12bit_hanning = compute_snr_with_hanning(input_signal_noisy_30, quantized_signal_noisy_30_12bit, fs)
snr_noisy_100_12bit_hanning = compute_snr_with_hanning(input_signal_noisy_100, quantized_signal_noisy_100_12bit, fs)

# 결과 출력
print("E) Noisy Signal Quantization Results (SNR = 38 dB):")
print(f"6-bit SNR (30 periods) before noise: {snr_30_6bit:.2f} dB, after noise: {snr_noisy_30_6bit:.2f} dB")
print(f"6-bit SNR (100 periods) before noise: {snr_100_6bit:.2f} dB, after noise: {snr_noisy_100_6bit:.2f} dB")
print(f"12-bit SNR (30 periods) before noise: {snr_30_12bit:.2f} dB, after noise: {snr_noisy_30_12bit:.2f} dB")
print(f"12-bit SNR (100 periods) before noise: {snr_100_12bit:.2f} dB, after noise: {snr_noisy_100_12bit:.2f} dB")
print(f"6-bit SNR (30 periods) after Hanning: {snr_30_6bit_hanning:.2f} dB, after noise + Hanning: {snr_noisy_30_6bit_hanning:.2f} dB")
print(f"6-bit SNR (100 periods) after Hanning: {snr_100_6bit_hanning:.2f} dB, after noise + Hanning: {snr_noisy_100_6bit_hanning:.2f} dB")
print(f"12-bit SNR (30 periods) after Hanning: {snr_30_12bit_hanning:.2f} dB, after noise + Hanning: {snr_noisy_30_12bit_hanning:.2f} dB")
print(f"12-bit SNR (100 periods) after Hanning: {snr_100_12bit_hanning:.2f} dB, after noise + Hanning: {snr_noisy_100_12bit_hanning:.2f} dB")