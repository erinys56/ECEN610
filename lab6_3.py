import numpy as np
import matplotlib.pyplot as plt

# Simulation Parameters
N = 100_000                      # number of samples
fs = 500e6                       # sampling rate (500 MS/s)
fin = 200e6                      # input tone frequency (200 MHz, 40k cycles in 100k samples)
t = np.arange(N) / fs
input_signal = 0.25 * np.sin(2*np.pi*fin*t)   # 0.25 V amplitude sine wave

# Pipeline ADC Configuration
Stages = 6
A = 6.4                          # finite OTA open-loop gain
offset_ota = 0.185               # OTA output DC offset (Volts per stage)
comp_offset = 0.2                # comparator threshold offset (Volts per stage)
cap_mismatch_std = 0.22          # std-dev of capacitor mismatch (22%)

# Ideal 2.5-bit stage thresholds and DAC levels
ideal_thresholds = np.array([-0.375, -0.125, 0.125, 0.375])    # decision boundaries
ideal_dac_levels = np.array([-0.5, -0.25, 0.0, 0.25, 0.5])     # sub-ADC output levels for codes -2, -1, 0, +1, +2

# Set a fixed random seed for repeatability of the static mismatches
np.random.seed(0)
mismatch = np.random.normal(0, cap_mismatch_std, 5)   # random gain errors for stages 1-5
g_ideal = 4.0 * (1 + mismatch)                       # ideal MDAC gains (nominal 4) with mismatch

# Pipeline ADC simulation: capture each stage's digital output b[i]
b = {i: np.zeros(N, dtype=int) for i in range(1, Stages+1)}
Vin = input_signal.copy()
for i in range(1, Stages+1):
    # Comparator thresholds for this stage (including offset)
    thr = ideal_thresholds + comp_offset
    # Determine decision index 0–4 (for 5 levels) based on thresholds
    j = np.searchsorted(thr, Vin, side='right')
    j = np.clip(j, 0, 4)
    b[i] = j - 2   # convert index to code -2..+2
    # Compute analog residue for next stage (if not last stage)
    if i < Stages:
        # Effective closed-loop gain with finite op-amp gain A
        g_eff = (A * g_ideal[i-1]) / (A + g_ideal[i-1])
        # Residue = amplified (input - DAC_level) + OTA offset
        Vres = g_eff * (Vin - ideal_dac_levels[j]) + offset_ota
        Vin = Vres

# Reconstruct the ADC output (uncalibrated) assuming ideal bit weights
reconstructed_analog = (b[1]*0.25 + b[2]*0.25/4 + b[3]*0.25/16 + 
                        b[4]*0.25/64 + b[5]*0.25/256 + b[6]*0.25/1024)
# Convert to digital code (13-bit range 0 to 8191 corresponding to -0.5 to +0.5 V)
max_code = 2**13 - 1
adc_code_uncal = np.round((reconstructed_analog + 0.5) * max_code).astype(int)
adc_code_uncal = np.clip(adc_code_uncal, 0, max_code)

# Compute SNDR (Signal-to-Noise+Distortion) via FFT
def compute_sndr(code, fs, fin):
    code = code - np.mean(code)           # remove DC
    N = len(code)
    X = np.fft.fft(code)
    P = np.abs(X)**2
    k = int(np.round(fin/fs * N))        # index of the fundamental bin
    total_power = np.sum(P)
    fund_power = P[k] + P[-k]            # include both positive and negative frequency components
    noise_dist_power = total_power - fund_power - P[0]  # exclude DC and fundamental
    return 10 * np.log10(fund_power / noise_dist_power)

SNDR_before = compute_sndr(adc_code_uncal, fs, fin)
print(f"SNDR before calibration: {SNDR_before:.2f} dB")

# LMS Calibration – 6 weights (w0 + w1..w5 for stages 1–5)
mu = 0.5
w = np.zeros(6)               # initial weights [w0, w1, ..., w5]
w_history = np.zeros((N, 6))  # record weight evolution
error_sq = np.zeros(N)        # record squared error for analysis

for n in range(N):
    # Form input vector [1, b1, b2, b3, b4, b5] for this sample
    x_n = np.array([1, b[1][n], b[2][n], b[3][n], b[4][n], b[5][n]], dtype=float)
    y_n = np.dot(w, x_n)              # LMS filter output (predicted analog input)
    d_n = input_signal[n]             # desired signal (actual analog input at that sample)
    e_n = d_n - y_n                   # instantaneous error
    # Normalized LMS update: scale by 1/||x_n||^2 for better conditioning
    norm_factor = np.dot(x_n, x_n)
    w += (mu / norm_factor) * e_n * x_n
    # Record values
    w_history[n] = w
    error_sq[n] = e_n**2

w_final = w.copy()
print("Final LMS weights:", w_final)

# Apply final calibration to the captured data
# Use calibrated weights for stages 1-5, and assume ideal weight for stage 6
y_correction = (w_final[0] + w_final[1]*b[1] + w_final[2]*b[2] + 
                w_final[3]*b[3] + w_final[4]*b[4] + w_final[5]*b[5])
# Stage 6 ideal contribution (0.25/1024 per code step)
y_corrected_analog = y_correction + (0.25/1024) * b[6]
# Convert corrected analog estimate to digital code
adc_code_calibrated = np.round((y_corrected_analog + 0.5) * max_code).astype(int)
adc_code_calibrated = np.clip(adc_code_calibrated, 0, max_code)

SNDR_after = compute_sndr(adc_code_calibrated, fs, fin)
print(f"SNDR after calibration:  {SNDR_after:.2f} dB")

# Plot LMS weight convergence
plt.figure(figsize=(6,4))
for i, wi in enumerate(["w0","w1","w2","w3","w4","w5"]):
    plt.plot(w_history[:, i], label=wi)
plt.title("LMS Weight Convergence")
plt.xlabel("Iteration")
plt.ylabel("Weight Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot moving average of squared error to visualize convergence
window = 1000
mse = np.convolve(error_sq, np.ones(window)/window, mode='valid')
plt.figure(figsize=(6,4))
plt.plot(mse, color='purple')
plt.title(f"Moving Average of Squared Error (window={window})")
plt.xlabel("Iteration")
plt.ylabel("Mean Squared Error")
plt.grid(True)
plt.tight_layout()
plt.show()
