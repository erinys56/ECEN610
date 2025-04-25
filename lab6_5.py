import numpy as np
import matplotlib.pyplot as plt

# --- Signal generation ---
Fs = 500e6
fin = 200e6
N = 100000
t = np.arange(N)
A = 0.5
Vin_clean = A * np.sin(2*np.pi*fin/Fs * t)
signal_power = (A**2)/2
noise_var = signal_power / (10**(80/10))
noise = np.sqrt(noise_var) * np.random.randn(N)
Vin = Vin_clean + noise

# --- Pipeline ADC model (6 stages, 2.5-bit each) ---
def simulate_pipeline_adc(Vin, errors, nonlin):
    op_gain = errors.get('op_gain', np.inf)
    op_offset = errors.get('op_offset', 0.0)
    cap_m = errors.get('cap_mismatch', 0.0)
    comp_off = errors.get('comp_offset', 0.0)
    if np.isfinite(op_gain):
        gain_factor = op_gain/(op_gain+1)
    else:
        gain_factor = 1.0
    mm = 1 + cap_m
    a2 = nonlin.get('a2',0.0); a3 = nonlin.get('a3',0.0)
    a4 = nonlin.get('a4',0.0); a5 = nonlin.get('a5',0.0)
    
    X1 = Vin + comp_off
    d1 = np.floor((X1 + 0.5)/0.125).astype(int)
    d1 = np.clip(d1, 0, 7)
    dac1 = (d1 - 3.5)*0.125
    res1_lin = 4.0*(Vin - dac1)*gain_factor*mm + op_offset
    res1 = res1_lin + a2*res1_lin**2 + a3*res1_lin**3 + a4*res1_lin**4 + a5*res1_lin**5
    
    X2 = res1 + comp_off
    d2 = np.floor((X2 + 0.25)/0.0625).astype(int)
    d2 = np.clip(d2, 0, 7)
    dac2 = (d2 - 3.5)*0.0625
    res2_lin = 4.0*(res1 - dac2)*gain_factor*mm + op_offset
    res2 = res2_lin + a2*res2_lin**2 + a3*res2_lin**3 + a4*res2_lin**4 + a5*res2_lin**5

    X3 = res2 + comp_off
    d3 = np.floor((X3 + 0.125)/0.03125).astype(int)
    d3 = np.clip(d3, 0, 7)
    dac3 = (d3 - 3.5)*0.03125
    res3_lin = 4.0*(res2 - dac3)*gain_factor*mm + op_offset
    res3 = res3_lin + a2*res3_lin**2 + a3*res3_lin**3 + a4*res3_lin**4 + a5*res3_lin**5

    X4 = res3 + comp_off
    d4 = np.floor((X4 + 0.0625)/0.015625).astype(int)
    d4 = np.clip(d4, 0, 7)
    dac4 = (d4 - 3.5)*0.015625
    res4_lin = 4.0*(res3 - dac4)*gain_factor*mm + op_offset
    res4 = res4_lin + a2*res4_lin**2 + a3*res4_lin**3 + a4*res4_lin**4 + a5*res4_lin**5

    X5 = res4
    d5 = np.floor((X5 + 0.03125)/0.0078125).astype(int)
    d5 = np.clip(d5, 0, 7)
    dac5 = (d5 - 3.5)*0.0078125
    res5 = 4.0*(res4 - dac5)
    
    X6 = res5
    d6 = np.floor((X6 + 0.015625)/0.00390625).astype(int)
    d6 = np.clip(d6, 0, 7)

    return d1, d2, d3, d4, d5, d6

# Error settings
stage_errors = {
    'op_gain': 1000.0,
    'op_offset': 0.005,
    'cap_mismatch': 0.01,
    'comp_offset': 0.005
}
nonlin = {'a2':0.2, 'a3':0.1, 'a4':0.05, 'a5':0.02}

# Simulate ADC
d1, d2, d3, d4, d5, d6 = simulate_pipeline_adc(Vin, stage_errors, nonlin)

# Reconstruct uncalibrated output
w = np.array([0.125, 0.03125, 0.0078125, 0.001953125, 0.00048828125, 0.0001220703125])
codes = np.vstack([d1,d2,d3,d4,d5,d6]).T
analog_out = np.dot(codes - 3.5, w)

# SNDR calculation
def compute_sndr(ref_sig, out_sig, Fs, fin):
    y = out_sig - np.mean(out_sig)
    n = np.arange(len(y))
    sinw = np.sin(2*np.pi*fin/Fs * n)
    cosw = np.cos(2*np.pi*fin/Fs * n)
    a = 2/len(y) * np.dot(y, sinw)
    b = 2/len(y) * np.dot(y, cosw)
    fund_power = (a*a + b*b)/2
    total_power = np.var(y)
    noise_power = total_power - fund_power
    if noise_power <= 0:
        return np.inf
    return 10*np.log10(fund_power/noise_power)

SNDR_before = compute_sndr(Vin, analog_out, Fs, fin)
print(f"SNDR before calibration: {SNDR_before:.2f} dB")

# LMS calibration
d_list = [d1, d2, d3, d4]
d_norm = [(d - 3.5)/3.5 for d in d_list]
w_lin = np.array([0.4375, 0.109375, 0.02734375, 0.0068359375])
initial_w = np.zeros(21)
initial_w[:4] = w_lin
mu = 0.01

results = {}
decimations = [10, 100, 1000, 10000]
for decim in decimations:
    wts = initial_w.copy()
    for n in range(0, N, decim):
        if n >= N: break
        f = []
        for k in range(4):
            dn = d_norm[k][n]
            f.extend([dn, dn**2, dn**3, dn**4, dn**5])
        f.append(1.0)
        f = np.array(f)
        e = Vin[n] - np.dot(wts, f)
        wts += mu * e * f
    y_calib = np.zeros(N)
    for i in range(N):
        f = []
        for k in range(4):
            dn = d_norm[k][i]
            f.extend([dn, dn**2, dn**3, dn**4, dn**5])
        f.append(1.0)
        y_calib[i] = np.dot(wts, f)
    SNDR_after = compute_sndr(Vin, y_calib, Fs, fin)
    results[decim] = SNDR_after
    print(f"Decimation {decim}: SNDR after calib = {SNDR_after:.2f} dB")

# Full-rate LMS (decimation 1)
wts = initial_w.copy()
for n in range(N):
    f = []
    for k in range(4):
        dn = d_norm[k][n]
        f.extend([dn, dn**2, dn**3, dn**4, dn**5])
    f.append(1.0)
    f = np.array(f)
    e = Vin[n] - np.dot(wts, f)
    wts += mu * e * f

y_calib_full = np.zeros(N)
for i in range(N):
    f = []
    for k in range(4):
        dn = d_norm[k][i]
        f.extend([dn, dn**2, dn**3, dn**4, dn**5])
    f.append(1.0)
    y_calib_full[i] = np.dot(wts, f)

SNDR_dec1 = compute_sndr(Vin, y_calib_full, Fs, fin)
print(f"Decimation 1: SNDR after calib = {SNDR_dec1:.2f} dB")

# 결과 통합
decimations.append(1)
snr_values = [results[d] if d in results else SNDR_dec1 for d in decimations]

# === 추가 플롯 ===
plt.figure(figsize=(8,5))
plt.plot(decimations, snr_values, marker='o')
plt.xscale('log')
plt.xlabel('Decimation Factor (log scale)')
plt.ylabel('SNDR after Calibration (dB)')
plt.title('SNDR vs LMS Decimation Factor')
plt.grid(True, which='both')
plt.tight_layout()
plt.show()
