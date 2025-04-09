import numpy as np
import matplotlib.pyplot as plt


Vref = 1.0
offset = Vref / 8
fs = 10000
f_in = 100
t = np.arange(0, 1, 1/fs)
Vin = 0.9 * Vref * np.sin(2 * np.pi * f_in * t)



def quantize_2bit_code(x, offset=0):
    thresholds = np.array([-0.5, 0.0, 0.5]) * Vref + offset
    return np.digitize(x, thresholds)  # returns 0~3

def quantize_2_5bit_no_redun(x, offset=0):
    thresholds = np.linspace(-0.75, 0.75, 4) + offset
    return np.digitize(x, thresholds)  # returns 0~4

def quantize_2_5bit_with_redun_codes(x, offset=0):
    thresholds = np.linspace(-0.875, 0.875, 7) + offset
    return np.digitize(x, thresholds)  # returns 0~7


def dac_output(code, bits):
    if bits == 2:
        levels = np.linspace(-1.0, 1.0, 4)
    elif bits == 2.5 and len(np.unique(code)) <= 5:
        levels = np.linspace(-1.0, 1.0, 5)
    else:
        levels = np.linspace(-1.0, 1.0, 8)
    return levels[code]


def pipeline_residue_output(x, offset=0):
    code = quantize_2_5bit_with_redun_codes(x, offset)
    dac = dac_output(code, bits=2.5)
    residue = (x - dac) * 4  # G=4
    return code, residue, dac


def compute_snr(signal, quantized):
    error = signal - quantized
    signal_power = np.mean(signal**2)
    noise_power = np.mean(error**2)
    return 10 * np.log10(signal_power / noise_power)


x = np.linspace(-1.0, 1.0, 1000)

q2_code = quantize_2bit_code(x)
q2_5_no_code = quantize_2_5bit_no_redun(x)
q2_5_redun_code = quantize_2_5bit_with_redun_codes(x)

plt.figure(figsize=(10, 6))
plt.plot(x, q2_code, label="2-bit", linewidth=2)
plt.plot(x, q2_5_no_code, label="2.5-bit (No Redundancy)", linewidth=2)
plt.plot(x, q2_5_redun_code, label="2.5-bit (With Redundancy)", linewidth=2)
plt.title("Transfer Functions")
plt.xlabel("Input Voltage")
plt.ylabel("Quantized Output Code")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()




q2_code = quantize_2bit_code(Vin, offset)
q2_output = dac_output(q2_code, bits=2)
snr_2bit = compute_snr(Vin, q2_output)


q2_5_no_code = quantize_2_5bit_no_redun(Vin, offset)
q2_5_no_output = dac_output(q2_5_no_code, bits=2.5)
snr_2_5_no = compute_snr(Vin, q2_5_no_output)


q2_5_redun_code, _, q2_5_redun_output = pipeline_residue_output(Vin, offset)
snr_2_5_redun = compute_snr(Vin, q2_5_redun_output)


plt.figure(figsize=(12, 9))

plt.subplot(3,1,1)
plt.plot(t[:500], Vin[:500], label="Input", alpha=0.6)
plt.plot(t[:500], q2_output[:500], label="Quantized Output")
plt.title(f"2-bit Output with Offset, SNR = {snr_2bit:.2f} dB")
plt.grid(); plt.legend()

plt.subplot(3,1,2)
plt.plot(t[:500], Vin[:500], label="Input", alpha=0.6)
plt.plot(t[:500], q2_5_no_output[:500], label="Quantized Output")
plt.title(f"2.5-bit (No Redundancy) Output with Offset, SNR = {snr_2_5_no:.2f} dB")
plt.grid(); plt.legend()

plt.subplot(3,1,3)
plt.plot(t[:500], Vin[:500], label="Input", alpha=0.6)
plt.plot(t[:500], q2_5_redun_output[:500], label="Quantized Output")
plt.title(f"2.5-bit (With Redundancy) Output with Offset, SNR = {snr_2_5_redun:.2f} dB")
plt.grid(); plt.legend()

plt.tight_layout()
plt.show()
