# === Split Pipelined-SAR ADC + Neural Network Distortion Correction ===

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# === 1. setting ===
N_bits = 6
N_samples = 1000
t = np.linspace(0, 1, N_samples)
Vin = 0.8 * np.sin(2 * np.pi * 5 * t)  

# === 2. Coarse ADC ===
levels = np.linspace(-1, 1, 2**N_bits)
Vin_q = np.digitize(Vin, levels) - 1
Vin_q = np.clip(Vin_q, 0, len(levels) - 1)
Vcoarse = levels[Vin_q]

# === 3. Fine ADCs A/B ===
def generate_bit_vector(signal, threshold_levels):
    return np.where(signal[:, None] > threshold_levels, 1, -1)

thresholds = np.linspace(-1, 1, N_bits + 1)[1:-1]  # 5개 threshold
d_B = generate_bit_vector(Vin, thresholds)  
d_A = d_B.copy()

# === 4. Distortion  ===
distortion = 0.2 * np.sin(5 * np.pi * t)  
Vfine_A = Vcoarse + distortion
d_A_distorted = generate_bit_vector(Vfine_A, thresholds)

# === 5. Neural Network  ===
weights = np.linspace(1, 0.1, d_B.shape[1])  # comparator weight
x_base_A = d_A_distorted @ weights
x_base_B = d_B @ weights
error_signal = x_base_A - x_base_B

model = MLPRegressor(hidden_layer_sizes=(10,), activation='relu', max_iter=1000)
model.fit(d_A_distorted, -error_signal)  


x_corr = model.predict(d_A_distorted)
x_A_corrected = x_base_A + x_corr
x_out = 0.5 * (x_A_corrected + x_base_B)


mse_before = mean_squared_error(x_base_B, x_base_A)
mse_after = mean_squared_error(x_base_B, x_A_corrected)

plt.figure(figsize=(10, 5))
plt.plot(t, x_base_B, label='Channel B (Ideal)')
plt.plot(t, x_base_A, label='Channel A (Before Correction)', alpha=0.6)
plt.plot(t, x_A_corrected, label='Channel A (After Correction)', linestyle='--')
plt.title("ADC Output Before and After Correction")
plt.xlabel("Time")
plt.ylabel("Output")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === 8. SNR ===
def compute_snr(signal, noise):
    power_signal = np.mean(signal**2)
    power_noise = np.mean(noise**2)
    return 10 * np.log10(power_signal / power_noise)

noise_before = x_base_A - x_base_B
noise_after = x_A_corrected - x_base_B

snr_before = compute_snr(x_base_B, noise_before)
snr_after = compute_snr(x_base_B, noise_after)

print("MSE Before Correction:", mse_before)
print("MSE After Correction:", mse_after)
print("SNR Before Correction:", snr_before, "dB")
print("SNR After Correction:", snr_after, "dB")
