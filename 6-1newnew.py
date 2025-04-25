import numpy as np
import matplotlib.pyplot as plt

# === 기본 설정 ===
VFS = 1.0
Vref = VFS
N = 4096
Fs = 500e6
Ts = 1 / Fs
t = np.arange(N) * Ts
fin = 200e6
Vin = 0.25 * np.sin(2 * np.pi * fin * t)

# === DAC 중심값 기반 MDAC ===
ideal_dac_levels = np.array([-0.875, -0.5, 0.0, 0.5, 0.875]) * Vref

def mdac_with_dac_tracking(input_signal):
    thresholds = np.linspace(-Vref, Vref, 6)
    idx = np.digitize(input_signal, thresholds) - 1
    idx = np.clip(idx, 0, 4)
    residue = 4 * (input_signal - ideal_dac_levels[idx])
    return residue, ideal_dac_levels[idx]

# === 전체 pipeline: DAC 출력 수집 ===
def pipeline_adc_dac_outputs(Vin, stages=6):
    residue = Vin.copy()
    dac_outputs = []
    for _ in range(stages):
        residue, dac = mdac_with_dac_tracking(residue)
        dac_outputs.append(dac)
    return np.stack(dac_outputs, axis=1)  # shape: (N, stages)

# === reconstruction ===
dac_outputs = pipeline_adc_dac_outputs(Vin)
reconstructed_weighted = np.zeros_like(Vin)
for i in range(6):
    reconstructed_weighted += dac_outputs[:, i] / (4 ** i)

# === SNDR 계산 ===
def calculate_sndr(signal, reconstructed):
    noise = signal - reconstructed
    return 10 * np.log10(np.mean(signal**2) / np.mean(noise**2))

snr_weighted = calculate_sndr(Vin, reconstructed_weighted)
print(f"✅ SNDR (Analog Weighted Sum): {snr_weighted:.2f} dB")

# === 시각화 ===
plt.figure(figsize=(10, 4))
plt.plot(t[:500], Vin[:500], label='Input')
plt.plot(t[:500], reconstructed_weighted[:500], label='Reconstructed (Analog Weighted)', linestyle='--')
plt.title("Analog Weighted Sum - Pipeline ADC Output")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (V)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
