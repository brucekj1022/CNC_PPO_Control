import numpy as np
import control as ctrl
import matplotlib.pyplot as plt

# 載入 BUE_0-1Hz 實驗數據
data_path = '../ExperimentData/雙模型模擬/PRE_0~1Hz_無共振/simulation_data.npz'
data = np.load(data_path, allow_pickle=True)
CC_list = data['CC_list']  # shape: (50,)
ID_Plant_v2p = data['ID_Plant_v2p'].item()
actual_steps = int(data['actual_steps'])  # 50
print(f"載入數據: {actual_steps} 步")

# 顏色漸變 (藍→紅)
colors = plt.cm.coolwarm(np.linspace(0, 1, actual_steps))

# 繪圖 - 使用暫存.py格式
plt.figure(figsize=(12, 8))

for step in range(actual_steps):
    CC = CC_list[step]
    OLoop = ctrl.minreal(ctrl.ss2tf(CC * ID_Plant_v2p), tol=1e-3, verbose=False)
    mag, _, oma = ctrl.bode(OLoop, dB=True, omega_limits=[1e-2, 3e3], plot=False)
    plt.plot(oma, 20 * np.log10(mag), color=colors[step], linewidth=0.8)

plt.grid()
plt.xscale('log')
plt.xlim(1, 1e4)
plt.ylim(-70, 70)
plt.xlabel("Frequency (rad/s)", size=28)
plt.ylabel("Magnitude (dB)", size=28)
plt.title("PRE Open Loop", size=36)
plt.tick_params(axis='both', labelsize=20)
plt.show()
print("波德圖已保存至 Bode_All_Steps.png")