"""
CNC 工具箱 - 集合各種繪圖與分析工具
執行此腳本後選擇要執行的功能
"""

# ============================================================
# 以下為不完整程式片段，供複製使用
# ============================================================
'''畫隨機共振峰值(加上控制器) - 需要 CC, Plant, model_x 等變數，放在Training的片段中
import control as ctrl
plt.figure(figsize=(12, 6))
OLoop = ctrl.minreal(ctrl.ss2tf(CC * Plant['v2p']), tol=1e-3, verbose=False)
mag, _, oma = ctrl.bode(OLoop, dB=True, omega_limits=[1e-2, 3e3], plot=False)
plt.plot(oma, 20 * np.log10(mag), color='b', linewidth=2)
for i in range(100):
    Plant=model_x.uncertainty_Plant()
    OLoop = ctrl.minreal(ctrl.ss2tf(CC * Plant['v2p']), tol=1e-3, verbose=False)
    mag, _, oma = ctrl.bode(OLoop, dB=True, omega_limits=[1e-2, 3e3], plot=False)
    plt.plot(oma, 20 * np.log10(mag), color='r', linewidth=0.3)
plt.grid()
plt.xscale('log')
plt.xlim(1, 1e4)
plt.ylim(-70, 70)
plt.xlabel("Frequency (rad/s)", size=14)
plt.ylabel("Magnitude (dB)", size=14)
plt.title("Perturbed Resonant Ensemble", size=18)
plt.show()
'''

'''轉出CC - 需要 CC 變數，放在 Training 的片段中
CC_tf = ctrl.ss2tf(CC)
den = np.array(CC_tf.den[0][0], dtype=np.float32)
cdl=len(den)#controller_data_len
num = np.array(CC_tf.num[0][0], dtype=np.float32)
num = np.pad(num, (0, cdl - len(num)), mode='constant')#補齊避免分子階數不足
'''

'''畫神經網路輸出 - 需要 FC 變數
fig, ax = plt.subplots(figsize=(6, 4))
# 轉換頻率從Hz到rad/s，增益轉換為dB
freq_rad = FC[:, 0] * 2 * np.pi  # Hz to rad/s
gain_dB = 20 * np.log10(FC[:, 1])  # 線性增益轉dB
# 畫藍色action點和線
ax.semilogx(freq_rad, gain_dB, 'b-o', markersize=6)
ax.set_xlabel('Frequency (rad/s)', fontsize=20)
ax.set_ylabel('Magnitude(dB)', fontsize=20)
ax.tick_params(axis='both', labelsize=16)
ax.set_xlim([1e-2, 1e4])
ax.set_ylim([-60, 80])
ax.grid(True, which='major', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.show()
'''

# ============================================================
# 以下為工具箱主程式
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
import scipy
import scipy.signal
import CNC

# ============================================================
# 功能函數
# ============================================================

def plot_id_uncertainty_bode():
    """畫 ID model / Uncertainty model 波德圖"""
    Ts = 0.001
    model_x = CNC.CNCModel('x', Ts)
    ID_Plant = model_x.ID_Plant()
    uncertainty_Plant = model_x.uncertainty_Plant()

    omega = np.logspace(np.log10(0.1), np.log10(3000), num=5000)

    # 大小波德圖
    plt.figure(figsize=(10, 8))

    mag1, phase1, omega1 = ctrl.bode(ID_Plant['v2p'], omega, dB=True, plot=False)
    plt.subplot(2, 1, 1)
    plt.semilogx(omega1, 20 * np.log10(mag1))
    plt.title('ID model', fontsize=16)
    plt.xlabel('Frequency [rad/s]', fontsize=14)
    plt.ylabel('Magnitude [dB]', fontsize=14)
    plt.grid(True, which='both', linestyle='--')

    mag2, phase2, omega2 = ctrl.bode(uncertainty_Plant['v2p'], omega, dB=True, plot=False)
    plt.subplot(2, 1, 2)
    plt.semilogx(omega2, 20 * np.log10(mag2))
    plt.title('Uncertainty model', fontsize=16)
    plt.xlabel('Frequency [rad/s]', fontsize=14)
    plt.ylabel('Magnitude [dB]', fontsize=14)
    plt.grid(True, which='both', linestyle='--')

    plt.subplots_adjust(hspace=0.5)
    plt.show()

    # 相位波德圖
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.semilogx(omega1, phase1 * (180 / np.pi))
    plt.title('ID model - Phase', fontsize=16)
    plt.xlabel('Frequency [rad/s]', fontsize=14)
    plt.ylabel('Phase [degrees]', fontsize=14)
    plt.grid(True, which='both', linestyle='--')

    plt.subplot(2, 1, 2)
    plt.semilogx(omega2, phase2 * (180 / np.pi))
    plt.title('Uncertainty model - Phase', fontsize=16)
    plt.xlabel('Frequency [rad/s]', fontsize=14)
    plt.ylabel('Phase [degrees]', fontsize=14)
    plt.grid(True, which='both', linestyle='--')

    plt.subplots_adjust(hspace=0.5)
    plt.show()


def plot_random_resonance_plant():
    """畫隨機共振峰值（純受控體，100次）"""
    Ts = 0.001
    model_x = CNC.CNCModel('x', Ts)
    
    plt.figure(figsize=(12, 8))
    plant = model_x.ID_Plant()
    mag, _, oma = ctrl.bode(plant['v2p'], dB=True, omega_limits=[1e-2, 3e3], plot=False)
    plt.plot(oma, 20 * np.log10(mag), color='b', linewidth=2, label='ID Plant')
    
    for i in range(100):
        Plant = model_x.uncertainty_Plant()
        mag, _, oma = ctrl.bode(Plant['v2p'], dB=True, omega_limits=[1e-2, 3e3], plot=False)
        plt.plot(oma, 20 * np.log10(mag), color='r', linewidth=0.3, alpha=0.5)
    
    plt.grid()
    plt.xscale('log')
    plt.xlim(1, 1e4)
    plt.ylim(-70, 70)
    plt.xlabel("Frequency (rad/s)", size=20)
    plt.ylabel("Magnitude (dB)", size=20)
    plt.title("Uncertainty Plant Ensemble (100 samples)", size=24)
    plt.tick_params(axis='both', labelsize=16)
    plt.legend(fontsize=14)
    plt.show()


def plot_chirp_input():
    """畫系統鑑別輸入(Chirp 信號）"""
    Ts = 0.001
    path_time = 5
    Magnitude = 100
    t = np.arange(0, path_time, Ts)

    inputdata = scipy.signal.chirp(t, f0=0, f1=50, t1=path_time, method='linear', phi=-90) * Magnitude

    plt.figure(figsize=(12, 6))
    plt.plot(t, inputdata)
    plt.title('Chirp Signal (0-50 Hz)', fontsize=20)
    plt.xlabel('Time (s)', fontsize=16)
    plt.ylabel('Magnitude (rpm)', fontsize=16)
    plt.tick_params(axis='both', labelsize=14)
    plt.grid(True)
    plt.show()


def plot_reference_path():
    """畫參考路徑"""
    Ts = 0.001
    path_model = CNC.PathModel(Ts)
    path_model.plot_path()


def plot_resonance_bounds():
    """隨機共振峰值上下界"""
    Ts = 0.001
    axis = 'x'

    cnc_model = CNC.CNCModel(axis, Ts)
    ID_plant = cnc_model.ID_Plant()
    v2p_ID = ID_plant['v2p']

    omega_nyquist = 500 * 2 * np.pi
    omega_range = np.logspace(0, np.log10(omega_nyquist), 1000)

    mag_ID, phase_ID, omega_out = ctrl.frequency_response(v2p_ID, omega_range)
    mag_ID_dB = 20 * np.log10(np.abs(mag_ID.flatten()))

    min_resonance_omega = 300
    max_resonance_omega = 1000

    omega_flat = omega_out.flatten()
    mask = (omega_flat >= min_resonance_omega) & (omega_flat <= max_resonance_omega)
    omega_bounds = omega_flat[mask]
    mag_ID_bounds = mag_ID_dB[mask]

    gain_lower = np.ones_like(omega_bounds)
    gain_upper = np.ones_like(omega_bounds)

    for i, omega in enumerate(omega_bounds):
        x = (omega - min_resonance_omega) / (max_resonance_omega - min_resonance_omega)
        gain_lower[i] = (5 * x * np.exp(2*x) + 2) / 2
        gain_upper[i] = (10 * x * np.exp(2*x) + 2) / 2

    gain_lower_dB = mag_ID_bounds + 20 * np.log10(gain_lower)
    gain_upper_dB = mag_ID_bounds + 20 * np.log10(gain_upper)

    plt.figure(figsize=(10, 6))
    plt.semilogx(omega_out.flatten(), mag_ID_dB, 'b-', linewidth=1.5, label='Nominal Plant')
    plt.semilogx(omega_bounds, gain_upper_dB, 'r:', linewidth=2, label='Upper Bound')
    plt.semilogx(omega_bounds, gain_lower_dB, 'g--', linewidth=1.5, label='Lower Bound')

    plt.axvline(x=300, color='k', linestyle='--', linewidth=1, alpha=0.7)
    plt.axvline(x=1000, color='k', linestyle='--', linewidth=1, alpha=0.7)

    plt.xlabel('Frequency (rad/s)', fontsize=16)
    plt.ylabel('Magnitude (dB)', fontsize=16)
    plt.title('Random Resonance Peak Bounds', fontsize=20)
    plt.legend(loc='best', fontsize=14)
    plt.grid(True, which='both', alpha=0.3)
    plt.xlim([100, omega_nyquist])
    plt.ylim([-40, 40])
    plt.tick_params(axis='both', labelsize=14)
    plt.tight_layout()
    plt.show()


def dynamic_fft_mask_animation():
    """路徑傅立葉使用動態時間遮罩測試（產生動畫）"""
    import os
    import imageio
    
    Ts = 0.001
    pdl = 300
    fft_limit_freq = 15
    path_min_freq = 0.2
    
    def path_FFT(path, path_index, prev_dominant_freq):
        min_freq = 0.2
        if prev_dominant_freq < min_freq:
            prev_dominant_freq = min_freq
        FFT_mask = int(1 / prev_dominant_freq / Ts * 2)
        N = int(100000 / fft_limit_freq)
        
        if path_index + FFT_mask > len(path):
            path_mask = path[-FFT_mask:]
        else:
            path_mask = path[path_index:path_index + FFT_mask]
        
        path_mask = path_mask - np.mean(path_mask)
        hanning_window = np.hanning(FFT_mask)
        windowed_path = hanning_window * path_mask
        yf = scipy.fft.fft(windowed_path, N)
        xf = scipy.fft.fftfreq(N, Ts)
        
        magnitude = np.abs(yf[:N // 2]) / FFT_mask
        xf = xf[:N // 2]
        mask = xf < fft_limit_freq
        magnitude = magnitude[mask]
        
        magnitude_min = np.min(magnitude)
        magnitude_max = np.max(magnitude)
        normalized_magnitude = (magnitude - magnitude_min) / (magnitude_max - magnitude_min)
        
        peaks, _ = scipy.signal.find_peaks(normalized_magnitude)
        peak_magnitudes = normalized_magnitude[peaks]
        threshold = 0.7
        valid_peaks = peaks[peak_magnitudes >= threshold]
        if len(valid_peaks) > 0:
            dominant_freq = xf[valid_peaks].min()
        else:
            dominant_index = np.argmax(normalized_magnitude)
            dominant_freq = xf[dominant_index]
        return normalized_magnitude, dominant_freq, FFT_mask
    
    pathmodel = CNC.PathModel(Ts)
    path = pathmodel.up_down_chirp()
    path_index = 0
    num_path_district = int((len(path) - path_index) / pdl)
    prev_dominant_freq = path_min_freq
    
    if not os.path.exists('frames'):
        os.makedirs('frames')
    
    for step in range(num_path_district):
        FFT_data, prev_dominant_freq, FFT_mask = path_FFT(path, path_index, prev_dominant_freq)
        if step * pdl + FFT_mask > len(path):
            mask_start = (len(path) - FFT_mask) * Ts
            mask_end = len(path) * Ts
        else:
            mask_start = step * pdl * Ts
            mask_end = (step * pdl + FFT_mask) * Ts
        
        plt.figure(figsize=(12, 8))
        t = np.arange(0, len(path) * Ts, Ts)
        plt.plot(t, path, zorder=1)
        plt.axvspan(mask_start, mask_end, color='red', alpha=0.3, label="FFT Mask Region", zorder=2)
        plt.xlabel("Time", fontsize=40)
        plt.ylabel("Magnitude", fontsize=40)
        plt.title(f'Dynamic Mask Step {step + 1}', fontsize=40)
        plt.tick_params(axis='both', labelsize=32)
        plt.tight_layout()
        plt.savefig(f'frames/frame_{step:03d}.png')
        plt.clf()
        path_index = path_index + pdl
    
    with imageio.get_writer('animation.mp4', fps=5, codec='libx264', quality=8) as writer:
        for i in range(num_path_district):
            image = imageio.v2.imread(f'frames/frame_{i:03d}.png')
            writer.append_data(image)
    print("動畫已儲存為 animation.mp4")


def plot_experiment_openloop():
    """載入實驗數據並繪製每個Step開迴路波德圖，主要查看OLoop的隨機性"""
    # 預設路徑
    default_path = '../ExperimentData/雙模型模擬/PRE_0~1Hz_無共振/simulation_data.npz'
    
    print(f"預設數據路徑: {default_path}")
    user_path = input("輸入新路徑或按 Enter 使用預設: ").strip()
    data_path = user_path if user_path else default_path
    
    try:
        data = np.load(data_path, allow_pickle=True)
        CC_list = data['CC_list']
        ID_Plant_v2p = data['ID_Plant_v2p'].item()
        actual_steps = int(data['actual_steps'])
        print(f"載入數據: {actual_steps} 步")
        
        # 顏色漸變 (藍→紅)
        colors = plt.cm.coolwarm(np.linspace(0, 1, actual_steps))
        
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
        plt.title("Open Loop Bode Plot", size=36)
        plt.tick_params(axis='both', labelsize=20)
        plt.show()
    except FileNotFoundError:
        print(f"找不到檔案: {data_path}")
    except Exception as e:
        print(f"載入失敗: {e}")


def generate_test_controller():
    """產生測試控制器：中央控制器 + 二階共振系統，用於 Runtime.py"""
    import argparse
    
    np.set_printoptions(precision=15, suppress=True)
    
    # CNC 參數
    Ts = 0.001
    x_polegain = 0.4352
    numFC = 14
    num_low_freq_FC = 3
    pdl = 300
    
    # 預設共振參數
    print("=" * 60)
    print("測試控制器產生器")
    print("=" * 60)
    print("\n共振參數設定:")
    
    omega_input = input("  共振頻率 omega (rad/s) [預設 800]: ").strip()
    omega = int(omega_input) if omega_input else 800
    
    zeta_input = input("  阻尼比 zeta [預設 0.05]: ").strip()
    zeta = float(zeta_input) if zeta_input else 0.05
    
    gain_input = input("  峰值增益 gain (分子zeta倍數) [預設 12]: ").strip()
    gain = int(gain_input) if gain_input else 12
    
    # 創建 argparse namespace
    class CNC_parameter:
        Lq = 10
        w_sumError = 1e+3
        w_FCfreq = 4e+3
        w_Wgc = 1e+3
        w_earlyTrain = 5e-3
    
    # 創建實例
    model_x = CNC.CNCModel('x', Ts)
    path_model = CNC.PathModel(Ts)
    ID_Plant = model_x.ID_Plant()
    testpath = path_model.test_path()
    
    costfunction_x = CNC.Costfunction(CNC_parameter, x_polegain, ID_Plant, testpath, pdl, numFC, num_low_freq_FC)
    
    # ===== 1. 取得中央控制器 (Q=0) =====
    CC_central = costfunction_x.LFTExpandedSS(np.zeros((CNC_parameter.Lq, 1)))
    print("\n中央控制器 CC_central 已產生")
    
    # ===== 2. 設定共振參數並創建二階系統 =====
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
    
    # ===== 5. 詢問是否繪圖 =====
    plot_choice = input("\n是否繪製驗證圖? (y/n) [預設 y]: ").strip().lower()
    if plot_choice != 'n':
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


# ============================================================
# 主選單
# ============================================================

MENU = {
    '1': ('畫 ID/Uncertainty model 波德圖', plot_id_uncertainty_bode),
    '2': ('畫隨機共振峰值（純受控體 100次）', plot_random_resonance_plant),
    '3': ('畫系統鑑別輸入（Chirp）', plot_chirp_input),
    '4': ('畫參考路徑', plot_reference_path),
    '5': ('隨機共振峰值上下界', plot_resonance_bounds),
    '6': ('動態FFT遮罩測試（產生動畫）', dynamic_fft_mask_animation),
    '7': ('載入實驗數據繪製開迴路波德圖', plot_experiment_openloop),
    '8': ('產生測試控制器（中央控制器+共振）', generate_test_controller),
}


def main():
    while True:
        print("\n" + "=" * 50)
        print("CNC 工具箱")
        print("=" * 50)
        for key, (name, _) in MENU.items():
            print(f"  [{key}] {name}")
        print("  [q] 離開")
        print("=" * 50)
        
        choice = input("請選擇功能: ").strip().lower()
        
        if choice == 'q':
            print("再見！")
            break
        elif choice in MENU:
            name, func = MENU[choice]
            print(f"\n執行: {name}\n")
            try:
                func()
            except Exception as e:
                print(f"執行錯誤: {e}")
        else:
            print("無效選擇，請重新輸入")


if __name__ == '__main__':
    main()




