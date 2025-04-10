import numpy as np
import matplotlib.pyplot as plt

# 주어진 DNL
dnl = np.array([0, -0.5, 0, 0.5, -1, 0.5, 0.5, 0])

# INL 계산
inl = np.cumsum(dnl)

# 이상적인 DAC 출력 (LSB 기준 누적값)
output_voltage = [0]
for i in range(1, len(dnl)):
    step = 1 + dnl[i]
    output_voltage.append(output_voltage[-1] + step)
output_voltage = np.array(output_voltage)

# Offset 및 Full-Scale Error 반영
offset_error = 0.5  # LSB
fullscale_error = 0.5  # LSB
scale = (7 + fullscale_error) / 7  # 이상적 full scale: 7 LSB → 실제 7.5 LSB
output_with_errors = output_voltage * scale + offset_error

# 코드 값
codes = np.arange(8)

# 그래프 그리기
plt.figure(figsize=(6, 4))
plt.plot(codes, output_with_errors, marker='o', label='Actual Transfer Curve')
plt.plot(codes, codes, linestyle='--', color='gray', label='Ideal Transfer Curve')
plt.title("Transfer Curve with DNL and Offset/FS Error")
plt.xlabel("Input Code")
plt.ylabel("Output Voltage (LSB)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
