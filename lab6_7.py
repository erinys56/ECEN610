import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
Fs = 500e6
N = 16384
t = np.arange(N)/Fs

# Generate 128 uniformly spaced BPSK tones from 0 to 200 MHz
n_tones = 128
indices = 51*np.arange(n_tones)
freqs = indices * Fs / N
bits = np.random.choice([-1,1], size=n_tones)

amps = 0.05
x = np.zeros(N)
for i, b in enumerate(bits):
    f = freqs[i]
    if f == 0:
        x += amps*b
    else:
        x += amps*b * np.cos(2*np.pi*f*t)

# Pipeline ADC parameters
stages = 6
G_ideal = 4.0
A_ops        = np.array([100.0]*stages)
G_effective  = G_ideal/(1 + G_ideal/A_ops)
cap_errs     = np.array([0.03, -0.02, 0.04, -0.01, 0.02, -0.03])
ota_offsets  = np.array([0.005, -0.007, 0.01, -0.004, 0.002, -0.009])
comp_offsets = np.array([0.010, -0.005, 0.008, -0.006, 0.005, -0.004])
nonlin_coef  = np.array([0.10, 0.05, 0.08, 0.04, 0.07, 0.09])

# Simulate pipeline ADC
Vin = x.copy()
stage_outputs = np.zeros((stages, N))
thresholds = np.array([-0.75, -0.25, 0.25, 0.75])
for i in range(stages):
    Vin_offset = Vin + comp_offsets[i]
    inds = np.digitize(Vin_offset, thresholds)
    codes = np.zeros(N, dtype=int)
    codes[inds == 0] = -2
    codes[inds == 1] = -1
    codes[inds == 2] =  0
    codes[inds == 3] =  1
    codes[inds >= 4] =  2
    stage_outputs[i,:] = codes
    
    Vdac = codes * 0.5 * (1 + cap_errs[i])
    resid = G_effective[i] * (Vin - Vdac)
    resid = resid + ota_offsets[i] + nonlin_coef[i] * (Vin - Vdac)**2
    Vin = resid

powers = np.array([4**(stages-1-i) for i in range(stages)])
digital_sum = np.dot(powers, stage_outputs)
analog_uncal = digital_sum / 2730.0

# FFT analysis of uncalibrated output
X_uncal = np.fft.fft(analog_uncal)
tone_amps_uncal = np.real(X_uncal[indices])
bits_out_uncal = np.sign(tone_amps_uncal)
bits_out_uncal[bits_out_uncal==0] = 1
bits_out_uncal[0] = -bits_out_uncal[0]

original_bits = bits.copy()
BER_before = np.mean(bits_out_uncal != original_bits)
MSE_before = np.mean((x - analog_uncal)**2)

print(f"Before calibration:  BER = {BER_before:.4f}, MSE = {MSE_before:.5e}")

# LMS calibration
X = np.vstack([stage_outputs[i,:] for i in range(5)] + [np.ones(N)]).T
Y = x
w, *_ = np.linalg.lstsq(X, Y, rcond=None)
Y_cal = X @ w
analog_cal = Y_cal

# FFT analysis of calibrated output
X_cal = np.fft.fft(analog_cal)
tone_amps_cal = np.real(X_cal[indices])
bits_out_cal = np.sign(tone_amps_cal)
bits_out_cal[bits_out_cal==0] = 1
BER_after = np.mean(bits_out_cal != original_bits)
MSE_after = np.mean((x - analog_cal)**2)

print(f"After calibration:   BER = {BER_after:.4f}, MSE = {MSE_after:.5e}")

# === 추가된 플롯 ===

# 1. Uncalibrated Spectrum
plt.figure(figsize=(10,4))
f_axis = np.fft.fftfreq(N, d=1/Fs)
plt.plot(f_axis[:N//2]/1e6, 20*np.log10(np.abs(X_uncal[:N//2])/np.max(np.abs(X_uncal))), label='Uncalibrated')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Magnitude (dBFS)')
plt.title('FFT Spectrum Before Calibration')
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Calibrated Spectrum
plt.figure(figsize=(10,4))
plt.plot(f_axis[:N//2]/1e6, 20*np.log10(np.abs(X_cal[:N//2])/np.max(np.abs(X_cal))), label='Calibrated', color='orange')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Magnitude (dBFS)')
plt.title('FFT Spectrum After Calibration')
plt.grid(True)
plt.tight_layout()
plt.show()
