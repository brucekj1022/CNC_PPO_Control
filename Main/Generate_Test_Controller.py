"""
產生測試控制器：中央控制器 + 二階共振系統
用於 Runtime.py 中的 X軸共振測試控制器(新)
"""
import argparse
import numpy as np
import control as ctrl
import CNC

np.set_printoptions(precision=15, suppress=True)

# CNC 參數
Ts = 0.001
x_polegain = 0.4352
numFC = 14
num_low_freq_FC = 3
pdl = 300

parser = argparse.ArgumentParser(description="CNC參數")
parser.add_argument('--Lq', type=int, default=10)
parser.add_argument('--w_sumError', type=float, default=1e+3)
parser.add_argument('--w_FCfreq', type=float, default=4e+3)
parser.add_argument('--w_Wgc', type=float, default=1e+3)
parser.add_argument('--w_earlyTrain', type=float, default=5e-3)
CNC_parameter = parser.parse_args()

# 創建實例
model_x = CNC.CNCModel('x', Ts)
path_model = CNC.PathModel(Ts)
ID_Plant = model_x.ID_Plant()
testpath = path_model.test_path()

costfunction_x = CNC.Costfunction(CNC_parameter, x_polegain, ID_Plant, testpath, pdl, numFC, num_low_freq_FC)

# ===== 1. 取得中央控制器 (Q=0) =====
CC_central = costfunction_x.LFTExpandedSS(np.zeros((CNC_parameter.Lq, 1)))
print("=" * 60)
print("中央控制器 CC_central 已產生")

# ===== 2. 設定共振參數並創建二階系統 =====
# 參考 CNC.py 中的共振設定方式
omega = 800       # 共振頻率 (rad/s)
zeta = 0.05       # 阻尼比 (越大共振越平緩)
gain = 12         # 峰值增益 (分子zeta的倍數)

# 二階共振系統 (連續時間)
# H(s) = (s^2 + gain*zeta*omega*s + omega^2) / (s^2 + 2*zeta*omega*s + omega^2)
resonance_tf_continuous = ctrl.TransferFunction(
    [1, gain * zeta * omega, omega**2], 
    [1, 2 * zeta * omega, omega**2]
)
# 離散化
resonance_tf = ctrl.sample_system(resonance_tf_continuous, Ts)

print(f"\n共振參數: omega={omega} rad/s, zeta={zeta}, gain={gain}")
print(f"共振二階系統 (離散):")
print(f"  分子: {resonance_tf.num[0][0]}")
print(f"  分母: {resonance_tf.den[0][0]}")

# ===== 3. 串接控制器 =====
CC_with_resonance = CC_central * resonance_tf
print("\n中央控制器已串接共振系統")

# ===== 4. 轉換為傳遞函數並輸出係數 =====
CC_tf = ctrl.ss2tf(CC_with_resonance)
num = np.array(CC_tf.num[0][0])
den = np.array(CC_tf.den[0][0])

print("\n" + "=" * 60)
print("串接後控制器的傳遞函數係數:")
print(f"分子 (num, {len(num)}個): {num}")
print(f"分母 (den, {len(den)}個): {den}")

# 對齊長度 (如果需要)
max_len = max(len(num), len(den))
num_padded = np.pad(num, (0, max_len - len(num)))
den_padded = np.pad(den, (0, max_len - len(den)))

# 合併成 Runtime.py 格式
X_resonance_new = np.concatenate([num_padded, den_padded])

print("\n" + "=" * 60)
print("複製以下內容到 Runtime.py:")
print("=" * 60)
print(f"#X軸本身機台共振測試控制器(新) omega={omega}, zeta={zeta}, gain={gain}")
print(f"X_resonance = {X_resonance_new.tolist()}")
print(f"CC_X_resonance = ctrl.tf2ss(ctrl.TransferFunction(X_resonance[:{max_len}], X_resonance[{max_len}:], Ts))")
print("=" * 60)

# ===== 5. 驗證 =====
print("\n驗證串接控制器的頻率響應...")
import matplotlib.pyplot as plt

# 1. 中央控制器
plt.figure(figsize=(12, 6))
mag_c, _, oma_c = ctrl.bode(ctrl.ss2tf(CC_central), dB=True, omega_limits=[1e-2, 3e3], plot=False)
plt.plot(oma_c, 20*np.log10(mag_c), color='b', linewidth=2)
plt.grid()
plt.xscale('log')
plt.xlim(1, 1e4)
plt.ylim(-70, 70)
plt.xlabel("Frequency (rad/s)", size=14)
plt.ylabel("Magnitude (dB)", size=14)
plt.title("Central Controller", size=18)
plt.show()

# 2. 中央控制器 + ID_Plant (開迴路)
plt.figure(figsize=(12, 6))
OLoop_central = ctrl.minreal(ctrl.ss2tf(CC_central * ID_Plant['v2p']), tol=1e-3, verbose=False)
mag_oc, _, oma_oc = ctrl.bode(OLoop_central, dB=True, omega_limits=[1e-2, 3e3], plot=False)
plt.plot(oma_oc, 20*np.log10(mag_oc), color='b', linewidth=2)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
plt.grid()
plt.xscale('log')
plt.xlim(1, 1e4)
plt.ylim(-70, 70)
plt.xlabel("Frequency (rad/s)", size=14)
plt.ylabel("Magnitude (dB)", size=14)
plt.title("Central Controller + ID_Plant (Open Loop)", size=18)
plt.show()

# 3. 新測試控制器 (中央控制器 + 共振)
plt.figure(figsize=(12, 6))
mag_r, _, oma_r = ctrl.bode(CC_tf, dB=True, omega_limits=[1e-2, 3e3], plot=False)
plt.plot(oma_r, 20*np.log10(mag_r), color='r', linewidth=2)
plt.axvline(x=omega, color='g', linestyle='--', alpha=0.7, label=f'Resonance @ {omega} rad/s')
plt.grid()
plt.xscale('log')
plt.xlim(1, 1e4)
plt.ylim(-70, 70)
plt.xlabel("Frequency (rad/s)", size=14)
plt.ylabel("Magnitude (dB)", size=14)
plt.title("Test Controller (Central + Resonance)", size=18)
plt.legend()
plt.show()

# 4. 新測試控制器 + ID_Plant (開迴路)
plt.figure(figsize=(12, 6))
OLoop_resonance = ctrl.minreal(ctrl.ss2tf(CC_with_resonance * ID_Plant['v2p']), tol=1e-3, verbose=False)
mag_or, _, oma_or = ctrl.bode(OLoop_resonance, dB=True, omega_limits=[1e-2, 3e3], plot=False)
plt.plot(oma_or, 20*np.log10(mag_or), color='r', linewidth=2)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
plt.axvline(x=omega, color='g', linestyle='--', alpha=0.7, label=f'Resonance @ {omega} rad/s')
plt.grid()
plt.xscale('log')
plt.xlim(1, 1e4)
plt.ylim(-70, 70)
plt.xlabel("Frequency (rad/s)", size=14)
plt.ylabel("Magnitude (dB)", size=14)
plt.title("Test Controller + ID_Plant (Open Loop)", size=18)
plt.legend()
plt.show()
