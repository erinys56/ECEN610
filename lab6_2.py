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
ideal_dac_levels = np.array([-0.875, -0.5, 0.0, 0.5, 0.875]) * Vref

# === SNDR 계산 ===
def calculate_sndr(signal, reconstructed):
    noise = signal - reconstructed
    return 10 * np.log10(np.mean(signal**2) / np.mean(noise**2))

# === 공통 출력 함수 ===
def plot_result(x, y, xlabel, ylabel, title):
    plt.figure(figsize=(10, 4))
    plt.plot(x, y, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === 각 스태틱 에러 모델 ===
def mdac_with_offset(input_signal, offset=0.0):
    thresholds = np.linspace(-Vref, Vref, 6)
    idx = np.digitize(input_signal, thresholds) - 1
    idx = np.clip(idx, 0, 4)
    dac = ideal_dac_levels[idx]
    residue = 4 * (input_signal - dac) + offset
    return residue, dac

def pipeline_adc_with_cumulative_offset(Vin, stages=6, offset_per_stage=0.0):
    residue = Vin.copy()
    dac_outputs = []
    for _ in range(stages):
        residue, dac = mdac_with_offset(residue, offset=offset_per_stage)
        dac_outputs.append(dac)
    return np.stack(dac_outputs, axis=1)

def mdac_pipeline_dac_sum(Vin, A_ota, stages=6):
    G_ideal = 4
    residue = Vin.copy()
    dac_outputs = []
    for _ in range(stages):
        thresholds = np.linspace(-Vref, Vref, 6)
        idx = np.digitize(residue, thresholds) - 1
        idx = np.clip(idx, 0, 4)
        dac = ideal_dac_levels[idx]
        dac_outputs.append(dac)
        G_real = (G_ideal * A_ota) / (A_ota + G_ideal)
        residue = G_real * (residue - dac)
    dac_outputs = np.stack(dac_outputs, axis=1)
    weights = np.array([1 / (4 ** i) for i in range(stages)])
    return np.dot(dac_outputs, weights)

def mdac_with_cap_mismatch(Vin, A_ota, mismatch_std=0.0, stages=6):
    G_ideal = 4
    residue = Vin.copy()
    dac_outputs = []
    for _ in range(stages):
        mismatch = np.random.normal(0, mismatch_std, size=ideal_dac_levels.shape)
        mismatched_dac_levels = ideal_dac_levels + mismatch
        thresholds = np.linspace(-Vref, Vref, 6)
        idx = np.digitize(residue, thresholds) - 1
        idx = np.clip(idx, 0, 4)
        dac = mismatched_dac_levels[idx]
        dac_outputs.append(dac)
        G_real = (G_ideal * A_ota) / (A_ota + G_ideal)
        residue = G_real * (residue - dac)
    dac_outputs = np.stack(dac_outputs, axis=1)
    weights = np.array([1 / (4 ** i) for i in range(stages)])
    return np.dot(dac_outputs, weights)

def mdac_with_comparator_offset(Vin, A_ota, comp_offset=0.0, stages=6):
    G_ideal = 4
    residue = Vin.copy()
    dac_outputs = []
    for _ in range(stages):
        thresholds = np.linspace(-Vref, Vref, 6) + comp_offset
        idx = np.digitize(residue, thresholds) - 1
        idx = np.clip(idx, 0, 4)
        dac = ideal_dac_levels[idx]
        dac_outputs.append(dac)
        G_real = (G_ideal * A_ota) / (A_ota + G_ideal)
        residue = G_real * (residue - dac)
    dac_outputs = np.stack(dac_outputs, axis=1)
    weights = np.array([1 / (4 ** i) for i in range(stages)])
    return np.dot(dac_outputs, weights)

def mdac_with_quadratic_nonlinear_gain(Vin, A0=2500, alpha=1.0, stages=6):
    G_ideal = 4
    residue = Vin.copy()
    dac_outputs = []
    for _ in range(stages):
        thresholds = np.linspace(-Vref, Vref, 6)
        idx = np.digitize(residue, thresholds) - 1
        idx = np.clip(idx, 0, 4)
        dac = ideal_dac_levels[idx]
        dac_outputs.append(dac)
        Vin_stage = residue - dac
        A_eff = A0 / (1 + (alpha * Vin_stage)**2)
        G_real = (G_ideal * A_eff) / (A_eff + G_ideal)
        residue = G_real * Vin_stage
    dac_outputs = np.stack(dac_outputs, axis=1)
    weights = np.array([1 / (4 ** i) for i in range(stages)])
    return np.dot(dac_outputs, weights)

def mdac_with_bandwidth_limit(Vin, A0=2500, f_bw=2e9, Fs=500e6, stages=6):
    G_ideal = 4
    residue = Vin.copy()
    dac_outputs = []
    alpha = np.exp(-2 * np.pi * f_bw / Fs)
    for _ in range(stages):
        thresholds = np.linspace(-Vref, Vref, 6)
        idx = np.digitize(residue, thresholds) - 1
        idx = np.clip(idx, 0, 4)
        dac = ideal_dac_levels[idx]
        dac_outputs.append(dac)
        Vin_stage = residue - dac
        A_eff = A0
        G_real = (G_ideal * A_eff) / (A_eff + G_ideal)
        ideal_output = G_real * Vin_stage
        residue = alpha * residue + (1 - alpha) * ideal_output
    dac_outputs = np.stack(dac_outputs, axis=1)
    weights = np.array([1 / (4 ** i) for i in range(stages)])
    return np.dot(dac_outputs, weights)
# === 실험 및 그래프 및 요약 ===
summary_results = {}

# 1. OTA Offset

offset = 0.0
step = 0.001
snr_offset_list = []
while True:
    dac_outputs = pipeline_adc_with_cumulative_offset(Vin, offset_per_stage=offset)
    rec = np.sum(dac_outputs / (4 ** np.arange(6)), axis=1)
    sndr = calculate_sndr(Vin, rec)
    snr_offset_list.append((offset, sndr))
    if sndr < 10:
        summary_results["OTA Offset (mV)"] = offset * 1e3
        break
    offset += step
plot_result([x[0]*1e3 for x in snr_offset_list], [x[1] for x in snr_offset_list],
            "OTA Offset per Stage (mV)", "SNDR (dB)", "SNDR vs OTA Offset")

# 2. OTA Gain

A_ota_list = np.logspace(np.log10(2500), np.log10(1), 60)
snr_gain_list = []
for A_ota in A_ota_list:
    rec = mdac_pipeline_dac_sum(Vin, A_ota)
    sndr = calculate_sndr(Vin, rec)
    snr_gain_list.append((A_ota, sndr))
    if sndr < 10:
        summary_results["OTA Gain"] = A_ota
        break
plot_result(*zip(*snr_gain_list),
            "A_ota (Finite OTA Gain)", "SNDR (dB)", "SNDR vs Finite OTA Gain")

# 3. Capacitor Mismatch

mismatch_std = 0.0
snr_mismatch_list = []
while mismatch_std <= 1.0:
    sndrs = [calculate_sndr(Vin, mdac_with_cap_mismatch(Vin, 2500, mismatch_std)) for _ in range(100)]
    avg_sndr = np.mean(sndrs)
    snr_mismatch_list.append((mismatch_std, avg_sndr))
    if avg_sndr < 10:
        summary_results["Cap Mismatch Std (%)"] = mismatch_std * 100
        break
    mismatch_std += 0.01
plot_result([x[0]*100 for x in snr_mismatch_list], [x[1] for x in snr_mismatch_list],
            "Mismatch Std (%)", "SNDR (dB)", "SNDR vs Cap Mismatch")

# 4. Comparator Offset

comp_offset = 0.0
snr_comp_list = []
while comp_offset <= 1.0:
    sndrs = [calculate_sndr(Vin, mdac_with_comparator_offset(Vin, 2500, comp_offset)) for _ in range(10)]
    avg_sndr = np.mean(sndrs)
    snr_comp_list.append((comp_offset, avg_sndr))
    if avg_sndr < 10:
        summary_results["Comparator Offset (mV)"] = comp_offset * 1e3
        break
    comp_offset += 0.001
plot_result([x[0]*1e3 for x in snr_comp_list], [x[1] for x in snr_comp_list],
            "Comparator Offset (mV)", "SNDR (dB)", "SNDR vs Comparator Offset")

# 5. Nonlinear Op-Amp Gain

alpha = 0.0
snr_nlq_list = []
while alpha <= 100:
    sndrs = [calculate_sndr(Vin, mdac_with_quadratic_nonlinear_gain(Vin, 2500, alpha)) for _ in range(10)]
    avg_sndr = np.mean(sndrs)
    snr_nlq_list.append((alpha, avg_sndr))
    if avg_sndr < 10:
        summary_results["Nonlinear α"] = alpha
        break
    alpha += 5.0
plot_result([x[0] for x in snr_nlq_list], [x[1] for x in snr_nlq_list],
            "Nonlinearity α", "SNDR (dB)", "SNDR vs Nonlinear Op-Amp")

# 6. Finite Bandwidth

f_bw_list = np.logspace(np.log10(6e8), np.log10(50e6), 100)
snr_bw_list = []
for f_bw in f_bw_list:
    rec = mdac_with_bandwidth_limit(Vin, A0=2500, f_bw=f_bw, Fs=Fs)
    sndr = calculate_sndr(Vin, rec)
    snr_bw_list.append((f_bw, sndr))
    if sndr < 10:
        summary_results["Bandwidth (MHz)"] = f_bw / 1e6
        break
plot_result([x[0]/1e6 for x in snr_bw_list], [x[1] for x in snr_bw_list],
            "Bandwidth (MHz)", "SNDR (dB)", "SNDR vs Finite Op-Amp Bandwidth")

# === 최종 요약 ===
print("\n SNDR dropped below 10 dB at:")
for k, v in summary_results.items():
    print(f"- {k}: {v:.2f}")
