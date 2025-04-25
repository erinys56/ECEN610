import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 100000                        # number of samples
fs = 500e6                        # sampling rate
fin = 200e6                       # input sine frequency (200 MHz)
t = np.arange(N) / fs
input_signal = 0.25 * np.sin(2*np.pi*fin*t)  # 0.25 V amplitude sine, zero-mean

# Stage specifications
num_stages = 6
threshold_nominal = np.array([-0.375, -0.125, 0.125, 0.375])  # ideal sub-ADC thresholds
np.random.seed(1)  # reproducible random offsets

# Generate stage error parameters (stages 1–5)
stages = []
for i in range(1, num_stages+1):
    if i < num_stages:
        comp_offsets = np.random.normal(0.0, 0.2, size=4)
        comp_offsets = np.clip(comp_offsets, -0.25, 0.25)
        thr_actual = np.sort(threshold_nominal + comp_offsets)
        mismatch = np.random.normal(0.0, 0.22)
        ideal_gain = 4.0
        gain_err = ideal_gain * (1 + mismatch)
        A = 6.4
        gain_actual = gain_err * (A/(A+1))
        ota_offset = (1 if np.random.rand()<0.5 else -1) * 0.185
        stages.append((thr_actual, gain_actual, ota_offset))
    else:
        stages.append((None, None, 0.0))

# Pipeline conversion
k_codes = np.zeros((num_stages, N))
analog_residues = np.zeros((num_stages+1, N))
analog_residues[0] = input_signal
for i in range(1, num_stages+1):
    thr, gain, offset = stages[i-1]
    prev = analog_residues[i-1]
    if i < num_stages:
        code_index = np.digitize(prev, thr)
        k = code_index - 2
        k_codes[i-1] = k
        analog_residues[i] = gain * (prev - 0.25*k) + offset
    else:
        resid5 = prev
        k6 = np.round(resid5 / 0.125)
        k6 = np.clip(k6, -4, 3)
        k_codes[i-1] = k6
        analog_residues[i] = resid5

# Uncalibrated output
w_ideal = np.array([0.0, 0.25, 0.0625, 0.015625, 0.00390625, 0.0009765625])
y_uncal = (w_ideal[1]*k_codes[0] + w_ideal[2]*k_codes[1] +
           w_ideal[3]*k_codes[2] + w_ideal[4]*k_codes[3] +
           w_ideal[5]*k_codes[4] + 0.000244140625*k_codes[5])
code_uncal = np.round(y_uncal * 8192)
code_uncal = np.clip(code_uncal, -4096, 4095).astype(int)

# NLMS adaptive calibration
X = np.vstack([np.ones(N), k_codes[0], k_codes[1], k_codes[2], k_codes[3], k_codes[4]]).T
d = input_signal.copy()
w = np.array([0.0, 0.25, 0.0625, 0.015625, 0.00390625, 0.0009765625])
mu = 0.5
delta = 1e-9

w_history = np.zeros((N, 6))   # weight tracking
e_sq_history = np.zeros(N)     # error square tracking

for n in range(N):
    x_n = X[n]
    y_n = np.dot(w, x_n) + 0.000244140625 * k_codes[5, n]
    e_n = d[n] - y_n
    norm_x = np.dot(x_n, x_n)
    w += (mu / (norm_x + delta)) * e_n * x_n
    w_history[n] = w
    e_sq_history[n] = e_n**2

w_final = w

# Apply final weights
y_cal = (w_final[0] + w_final[1]*k_codes[0] + w_final[2]*k_codes[1] +
         w_final[3]*k_codes[2] + w_final[4]*k_codes[3] + w_final[5]*k_codes[4] +
         0.000244140625 * k_codes[5])
code_cal = np.round(y_cal * 8192)
code_cal = np.clip(code_cal, -4096, 4095).astype(int)

# SNDR computation
def sndr(signal, reference):
    sig = signal - np.mean(signal)
    ref = reference - np.mean(reference)
    Psig = np.mean(ref**2)
    Perr = np.mean((signal - reference)**2)
    return 10*np.log10(Psig/Perr)

SNDR_before = sndr(code_uncal/8192.0, input_signal)
SNDR_after  = sndr(code_cal/8192.0, input_signal)

print("SNDR before calibration: %.2f dB" % SNDR_before)
print("SNDR after calibration:  %.2f dB" % SNDR_after)
print("Final NLMS weights:", w_final)

# === 추가된 플롯 ===

import matplotlib.pyplot as plt

# 1. LMS weight convergence plot
plt.figure(figsize=(8,5))
for i in range(6):
    plt.plot(w_history[:,i], label=f'w{i}')
plt.xlabel('Iteration')
plt.ylabel('Weight Value')
plt.title('NLMS Weight Convergence')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 2. Error decay plot (moving average)
window = 1000
moving_avg_error = np.convolve(e_sq_history, np.ones(window)/window, mode='valid')
plt.figure(figsize=(8,5))
plt.plot(moving_avg_error)
plt.xlabel('Iteration')
plt.ylabel('MSE (Moving Average)')
plt.title('Error Decay during NLMS Adaptation')
plt.grid(True)
plt.tight_layout()
plt.show()
