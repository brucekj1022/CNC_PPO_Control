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
    Plant=model_x.BUE_Plant()
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
import matplotlib
import matplotlib.pyplot as plt
import control as ctrl
import scipy
import scipy.signal
import tkinter as tk
from tkinter import filedialog
import os
import CNC

# ============================================================
# 繪圖參數(統一標準，與 Plot_Exp_Data.py 一致)
# ============================================================

# 圖片尺寸 (寬, 高) - 需被16整除以相容影片編碼
FIG_SIZE_SINGLE = (7.68, 5.76)    # 768x576
FIG_SIZE_WIDE = (11.52, 5.76)     # 1152x576
FIG_SIZE_MULTI = (10.24, 10.24)   # 1024x1024

# 字體大小設定
matplotlib.rcParams.update({
    'axes.titlesize': 24,     # 圖表標題
    'axes.labelsize': 20,     # 座標軸標籤
    'xtick.labelsize': 18,    # 刻度數字
    'ytick.labelsize': 18,    # 刻度數字
    'legend.fontsize': 18,    # 圖例
})

# ============================================================
# 功能函數
# ============================================================

def plot_plant_ensemble():
    """受控體波德圖繪製，可選 ID / Test / BUE / PRE 模型"""
    print("\n選擇模型類型:")
    print("  [1] ID Plant")
    print("  [2] Test Plant")
    print("  [3] BUE")
    print("  [4] PRE")
    print("  [b] 返回")
    model_choice = input("請選擇: ").strip().lower()
    if model_choice == 'b':
        return
    if model_choice not in ('1', '2', '3', '4'):
        print("無效選擇")
        return

    Ts = 0.001
    model_x = CNC.CNCModel('x', Ts)

    print("\n選擇轉移函數類型:")
    print("  [1] v2p (速度→位置)")
    print("  [2] v2v (速度→速度)")
    tf_choice = input("請選擇 [預設 2]: ").strip()
    tf_key = 'v2p' if tf_choice == '1' else 'v2v'

    plt.figure(figsize=FIG_SIZE_SINGLE)

    if model_choice == '1':
        plant = model_x.ID_Plant()
        mag, _, oma = ctrl.bode(plant[tf_key], dB=True, omega_limits=[1e-2, 3e3], plot=False)
        plt.plot(oma, 20 * np.log10(mag), color='b', linewidth=2, label='ID Plant')
        plt.title(f"ID Plant {tf_key}")

    elif model_choice == '2':
        id_plant = model_x.ID_Plant()
        mag, _, oma = ctrl.bode(id_plant[tf_key], dB=True, omega_limits=[1e-2, 3e3], plot=False)
        plt.plot(oma, 20 * np.log10(mag), color='b', linewidth=2, label='ID Plant')
        test_plant = model_x.test_Plant()
        mag, _, oma = ctrl.bode(test_plant[tf_key], dB=True, omega_limits=[1e-2, 3e3], plot=False)
        plt.plot(oma, 20 * np.log10(mag), color='r', linewidth=2, label='Test Plant')
        plt.title(f"Test Plant {tf_key}")

    else:
        count_input = input("抽取次數 [預設 100]: ").strip()
        try:
            count = int(count_input) if count_input else 100
            if count <= 0:
                raise ValueError
        except ValueError:
            print("無效輸入，使用預設值 100")
            count = 100

        model_name = "BUE" if model_choice == '3' else "PRE"
        get_plant = model_x.BUE_Plant if model_choice == '3' else model_x.PRE_Plant

        print(f"繪製 {model_name} 集合，共 {count} 次...")
        id_plant = model_x.ID_Plant()
        mag, _, oma = ctrl.bode(id_plant[tf_key], dB=True, omega_limits=[1e-2, 3e3], plot=False)
        plt.plot(oma, 20 * np.log10(mag), color='b', linewidth=2, label='ID Plant')
        for _ in range(count):
            plant = get_plant()
            mag, _, oma = ctrl.bode(plant[tf_key], dB=True, omega_limits=[1e-2, 3e3], plot=False)
            plt.plot(oma, 20 * np.log10(mag), color='r', linewidth=0.3, alpha=0.5)
        plt.title(f"{model_name} {tf_key} Ensemble ({count} samples)")

    plt.grid()
    plt.xscale('log')
    plt.xlim(1e-2, 1e4)
    plt.xlabel("Frequency (rad/s)")
    plt.ylabel("Magnitude (dB)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_or_export_path():
    """路徑繪圖與匯出工具 (支援選擇路徑類型、繪圖或匯出)"""
    import pandas as pd

    def _plot_single_path(t, path, title):
        plt.figure(figsize=FIG_SIZE_SINGLE)
        plt.plot(t[:len(path)], path, color='b')
        plt.title(title)
        plt.xlabel('Time (s)')
        plt.ylabel('Magnitude (mm)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    Ts = 0.001
    path_model = CNC.PathModel(Ts)
    
    # === 第一層選單：選擇路徑類型 ===
    print("\n選擇路徑類型:")
    print("  [1] test_path (0~1 Hz chirp)")
    print("  [2] test_path2 (0~8 Hz chirp)")
    print("  [3] training_path (20條混合路徑)")
    print("  [4] up_down_chirp (上下掃頻)")
    print("  [b] 返回")
    
    path_choice = input("請選擇: ").strip().lower()
    if path_choice == 'b':
        return
    
    # 取得路徑資料
    if path_choice == '1':
        paths = [path_model.test_path()]
        path_name = "test_path"
    elif path_choice == '2':
        paths = [path_model.test_path2()]
        path_name = "test_path2"
    elif path_choice == '3':
        paths = path_model.training_path()
        path_name = "training_path"
    elif path_choice == '4':
        paths = [path_model.up_down_chirp()]
        path_name = "up_down_chirp"
    else:
        print("無效選擇")
        return
    
    # === 第二層選單：選擇操作 ===
    print("\n選擇操作:")
    print("  [1] 繪圖")
    print("  [2] 匯出 TXT")
    print("  [3] 匯出 Excel")
    print("  [b] 返回")
    
    action_choice = input("請選擇: ").strip().lower()
    if action_choice == 'b':
        return
    
    t = np.arange(0, len(paths[0]) * Ts, Ts)
    
    if action_choice == '1':  # 繪圖模式
        if len(paths) > 1:  # training_path 多條路徑
            print("\n選擇要繪製的路徑:")
            print("  [0] 全部 (子圖排列)")
            for i in range(len(paths)):
                print(f"  [{i+1}] Path {i+1}")
            print("  [b] 返回")
            
            idx_choice = input("請選擇: ").strip().lower()
            if idx_choice == 'b':
                return
            elif idx_choice == '0':
                # 畫全部子圖
                n_paths = len(paths)
                cols = int(np.ceil(np.sqrt(n_paths)))
                rows = int(np.ceil(n_paths / cols))
                plt.figure(figsize=(4 * cols, 3 * rows))
                plt.suptitle('Training Paths', fontsize=24)
                for i, path in enumerate(paths):
                    ax = plt.subplot(rows, cols, i + 1)
                    ax.plot(t[:len(path)], path)
                    ax.grid(True)
                    ax.set_title(f"Path {i+1}", fontsize=18)
                    ax.tick_params(axis='both', labelsize=14)
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.show()
            else:
                try:
                    idx = int(idx_choice) - 1
                    if 0 <= idx < len(paths):
                        _plot_single_path(t, paths[idx], f"{path_name} [{idx+1}]")
                    else:
                        print("索引超出範圍")
                except ValueError:
                    print("無效輸入")
        else:
            _plot_single_path(t, paths[0], path_name)
    
    elif action_choice == '2':  # 匯出 TXT
        if len(paths) > 1:
            print("\n選擇要匯出的路徑:")
            print("  [0] 全部 (分別存檔)")
            for i in range(len(paths)):
                print(f"  [{i+1}] Path {i+1}")
            print("  [b] 返回")
            idx_choice = input("請選擇: ").strip().lower()
            if idx_choice == 'b':
                return
            elif idx_choice == '0':
                for i, path in enumerate(paths):
                    filename = f"{path_name}_{i+1}.txt"
                    np.savetxt(filename, path, delimiter=',', fmt='%.6f')
                    print(f"已匯出: {filename}")
            else:
                try:
                    idx = int(idx_choice) - 1
                    if 0 <= idx < len(paths):
                        filename = f"{path_name}_{idx+1}.txt"
                        np.savetxt(filename, paths[idx], delimiter=',', fmt='%.6f')
                        print(f"已匯出: {filename}")
                    else:
                        print("索引超出範圍")
                except ValueError:
                    print("無效輸入")
        else:
            filename = f"{path_name}.txt"
            np.savetxt(filename, paths[0], delimiter=',', fmt='%.6f')
            print(f"已匯出: {filename}")
    
    elif action_choice == '3':  # 匯出 Excel
        if len(paths) > 1:
            print("\n選擇要匯出的路徑:")
            print("  [0] 全部 (合併為單一檔案)")
            for i in range(len(paths)):
                print(f"  [{i+1}] Path {i+1}")
            print("  [b] 返回")
            idx_choice = input("請選擇: ").strip().lower()
            if idx_choice == 'b':
                return
            elif idx_choice == '0':
                # 所有路徑合併成一個 DataFrame
                data = {f"Path_{i+1}": path for i, path in enumerate(paths)}
                df = pd.DataFrame(data)
                filename = f"{path_name}_all.xlsx"
                df.to_excel(filename, index=False, engine='openpyxl')
                print(f"已匯出: {filename}")
            else:
                try:
                    idx = int(idx_choice) - 1
                    if 0 <= idx < len(paths):
                        df = pd.DataFrame(paths[idx], columns=['value'])
                        filename = f"{path_name}_{idx+1}.xlsx"
                        df.to_excel(filename, index=False, engine='openpyxl')
                        print(f"已匯出: {filename}")
                    else:
                        print("索引超出範圍")
                except ValueError:
                    print("無效輸入")
        else:
            df = pd.DataFrame(paths[0], columns=['value'])
            filename = f"{path_name}.xlsx"
            df.to_excel(filename, index=False, engine='openpyxl')
            print(f"已匯出: {filename}")
    else:
        print("無效選擇")


def plot_resonance_bounds():
    """隨機共振峰值上下界"""
    Ts = 0.001
    axis = 'x'

    cnc_model = CNC.CNCModel(axis, Ts)
    ID_plant = cnc_model.ID_Plant()

    print("\n選擇轉移函數類型:")
    print("  [1] v2p (速度→位置)")
    print("  [2] v2v (速度→速度)")
    tf_choice = input("請選擇 [預設 1]: ").strip()
    tf_key = 'v2v' if tf_choice == '2' else 'v2p'
    v2p_ID = ID_plant[tf_key]

    omega_nyquist = 500 * 2 * np.pi
    omega_range = np.linspace(100, omega_nyquist, 1000)

    mag_ID, _, omega_out = ctrl.bode(v2p_ID, omega_range, dB=True, plot=False)
    mag_ID_dB = 20 * np.log10(np.abs(np.array(mag_ID).flatten()))
    omega_flat = np.array(omega_out).flatten()

    min_resonance_omega = 300
    max_resonance_omega = 1000

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

    plt.figure(figsize=FIG_SIZE_SINGLE)
    plt.semilogx(omega_flat, mag_ID_dB, color='b', linewidth=1.5, label='Nominal Plant')
    plt.semilogx(omega_bounds, gain_upper_dB, color='r', linewidth=1.5, linestyle=':', label='Upper Bound')
    plt.semilogx(omega_bounds, gain_lower_dB, color='g', linewidth=1.5, linestyle='--', label='Lower Bound')

    plt.axvline(x=300, color='k', linestyle='--', linewidth=1, alpha=0.7)
    plt.axvline(x=1000, color='k', linestyle='--', linewidth=1, alpha=0.7)

    plt.xlabel('Frequency (rad/s)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Random Resonance Peak Bounds')
    plt.legend(loc='best')
    plt.grid(True, which='both', alpha=0.3)
    plt.xlim([100, omega_nyquist])
    plt.tight_layout()
    plt.show()


def dynamic_fft_mask_animation():
    """路徑傅立葉使用動態時間遮罩測試(產生動畫)"""
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
        
        plt.figure(figsize=FIG_SIZE_SINGLE)
        t = np.arange(0, len(path) * Ts, Ts)
        plt.plot(t, path, zorder=1)
        plt.axvspan(mask_start, mask_end, color='red', alpha=0.3, label="FFT Mask Region", zorder=2)
        plt.xlabel("Time")
        plt.ylabel("Magnitude")
        plt.title(f'Dynamic Mask Step {step + 1}')
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
    # 彈出檔案選擇視窗
    root = tk.Tk()
    root.withdraw()
    
    initial_dir = os.path.join("..", "ExperimentData")
    if not os.path.exists(initial_dir):
        initial_dir = ".."
    
    data_path = filedialog.askopenfilename(
        title="選擇實驗數據檔案 (runtime_data.npz 或 simulation_data.npz)",
        initialdir=initial_dir,
        filetypes=[("NumPy檔案", "*.npz"), ("所有檔案", "*.*")]
    )
    root.destroy()
    
    if not data_path:
        print("未選擇檔案")
        return
    
    print(f"選擇的檔案: {data_path}")
    
    try:
        data = np.load(data_path, allow_pickle=True)
        CC_list = data['CC_list']
        ID_Plant_v2p = data['ID_Plant_v2p'].item()
        actual_steps = int(data['actual_steps'])
        print(f"載入數據: {actual_steps} 步")
        
        plt.figure(figsize=FIG_SIZE_SINGLE)

        for step in range(actual_steps):
            CC = CC_list[step]
            OLoop = ctrl.minreal(ctrl.ss2tf(CC * ID_Plant_v2p), tol=1e-3, verbose=False)
            mag, _, oma = ctrl.bode(OLoop, dB=True, omega_limits=[1e-2, 3e3], plot=False)
            label = 'Controller' if step == 0 else None
            plt.plot(oma, 20 * np.log10(mag), color='b', linewidth=2, label=label)
        
        # 疊上中央控制器開路波德圖
        overlay_input = input("是否疊上中央控制器開路波德圖? (y/n) [預設 n]: ").strip().lower()
        if overlay_input == 'y':
            try:
                x_polegain = 0.4352
                numFC = 14
                num_low_freq_FC = 3
                pdl = 300
                class _CNC_parameter:
                    Lq = 10
                    w_sumError = 1e+3
                    w_FCfreq = 4e+3
                    w_Wgc = 1e+3
                    w_earlyTrain = 5e-3
                model_x = CNC.CNCModel('x', 0.001)
                ID_Plant = model_x.ID_Plant()
                testpath = CNC.PathModel(0.001).test_path()
                cf = CNC.Costfunction(_CNC_parameter, x_polegain, ID_Plant, testpath, pdl, numFC, num_low_freq_FC)
                CC_central = cf.LFTExpandedSS(np.zeros((_CNC_parameter.Lq, 1)))
                OLoop_central = ctrl.minreal(ctrl.ss2tf(CC_central * ID_Plant_v2p), tol=1e-3, verbose=False)
                mag_c, _, oma_c = ctrl.bode(OLoop_central, dB=True, omega_limits=[1e-2, 3e3], plot=False)
                plt.plot(oma_c, 20 * np.log10(mag_c), color='k', linewidth=2, label='Central Controller')
                plt.legend()
                print("中央控制器已疊上")
            except Exception as e:
                print(f"中央控制器載入失敗: {e}")

        plt.grid()
        plt.xscale('log')
        plt.xlim(1e-2, 1e4)
        plt.ylim(-70, 70)
        plt.xlabel("Frequency (rad/s)")
        plt.ylabel("Magnitude (dB)")
        plt.title("Open Loop Bode Plot")
        plt.tight_layout()
        output_path = os.path.join(os.path.dirname(data_path), 'openloop_bode.png')
        plt.savefig(output_path, dpi=150)
        print(f"圖片已儲存: {output_path}")
        plt.show()
    except FileNotFoundError:
        print(f"找不到檔案: {data_path}")
    except Exception as e:
        print(f"載入失敗: {e}")



def plot_experiment_OLoop_poles():
    """載入實驗數據並繪製閉迴路極點圖"""
    # 彈出檔案選擇視窗
    root = tk.Tk()
    root.withdraw()
    
    initial_dir = os.path.join("..", "ExperimentData")
    if not os.path.exists(initial_dir):
        initial_dir = ".."
    
    data_path = filedialog.askopenfilename(
        title="選擇實驗數據檔案 (runtime_data.npz 或 simulation_data.npz)",
        initialdir=initial_dir,
        filetypes=[("NumPy檔案", "*.npz"), ("所有檔案", "*.*")]
    )
    root.destroy()
    
    if not data_path:
        print("未選擇檔案")
        return
    
    print(f"選擇的檔案: {data_path}")
    
    try:
        data = np.load(data_path, allow_pickle=True)
        CC_list = data['CC_list']
        ID_Plant_v2p = data['ID_Plant_v2p'].item()
        Ts = float(data['Ts'])
        
        # 使用第一個控制器(假設所有步的控制器相同或只看第一個)
        CC = CC_list[0]
        
        # 閉迴路系統
        CL = ctrl.feedback(CC * ID_Plant_v2p, 1)
        poles_d = CL.poles()
        
        # 轉換到連續域(所有極點)
        poles_c = np.log(poles_d) / Ts
        
        print(f"Total poles: {len(poles_c)}")

        # 計算阻尼比（只對複數極點）
        complex_poles = poles_c[np.abs(poles_c.imag) > 1]
        if len(complex_poles) > 0:
            zeta = -complex_poles.real / np.abs(complex_poles)
            min_zeta = np.min(zeta)
            min_zeta_pole = complex_poles[np.argmin(zeta)]
            print(f"最小阻尼比 ζ = {min_zeta:.4f}  (極點: {min_zeta_pole.real:.0f} ± {abs(min_zeta_pole.imag):.0f}j rad/s)")

        # 選出離虛軸最近的前 5 個複數極點（取上半平面，避免共軛重複）
        upper_complex = complex_poles[complex_poles.imag > 0]
        if len(upper_complex) > 0:
            sorted_by_real = upper_complex[np.argsort(np.abs(upper_complex.real))]
            label_poles = sorted_by_real[:5]
        else:
            label_poles = np.array([])

        # 畫極點圖
        fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)

        # 畫所有極點
        ax.scatter(poles_c.real, poles_c.imag, s=100, c='blue', marker='x', linewidths=2)

        # 只標記離虛軸最近的複數極點（上下各標）
        for p in label_poles:
            for sign, pole in [(1, p), (-1, p.conjugate())]:
                label = f'({pole.real:.0f}, {pole.imag:.0f}j)'
                ax.annotate(label, (pole.real, pole.imag),
                            textcoords='offset points', xytext=(5, 5 * sign), fontsize=12)

        # 畫軸線
        ax.axvline(x=0, color='k', linestyle='-', linewidth=1)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=1)

        ax.set_xlabel('Real (1/s)')
        ax.set_ylabel('Imaginary (rad/s)')
        ax.set_title('Closed-Loop Poles (Test Controller)')
        ax.grid(True, alpha=0.3)

        # 自動調整範圍包含所有極點
        margin = 0.1
        real_min, real_max = poles_c.real.min(), poles_c.real.max()
        imag_min, imag_max = poles_c.imag.min(), poles_c.imag.max()
        real_range = max(real_max - real_min, 100)
        imag_range = max(imag_max - imag_min, 100)
        ax.set_xlim(real_min - margin * real_range - 500, real_max + margin * real_range + 500)
        ax.set_ylim(imag_min - margin * imag_range - 500, imag_max + margin * imag_range + 500)

        # 畫最小阻尼比線（set_xlim 之後才能拿到正確邊界）
        if len(complex_poles) > 0:
            pr, pi = min_zeta_pole.real, abs(min_zeta_pole.imag)
            slope = pi / pr  # pr < 0，slope < 0，往左上延伸
            x_left = ax.get_xlim()[0]
            y_at_left = slope * x_left
            ax.plot([0, x_left], [0, y_at_left], color='orange', linewidth=1.5)
            ax.plot([0, x_left], [0, -y_at_left], color='orange', linewidth=1.5)
            ax.text(0.02, 0.97, f'min ζ = {min_zeta:.4f}',
                    transform=ax.transAxes, fontsize=12, color='orange', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='orange', alpha=0.8))
        
        plt.tight_layout()
        output_path = os.path.join(os.path.dirname(data_path), 'closed_loop_poles.png')
        plt.savefig(output_path, dpi=150)
        print(f"圖片已儲存: {output_path}")
        plt.show()

    except FileNotFoundError:
        print(f"找不到檔案: {data_path}")
    except Exception as e:
        print(f"載入失敗: {e}")


def plot_resonance_spectrum():
    """載入實驗數據繪製誤差全段 FFT 頻譜分析"""
    import scipy.fft
    import scipy.signal

    root = tk.Tk()
    root.withdraw()
    initial_dir = os.path.join('..', 'ExperimentData')
    if not os.path.exists(initial_dir):
        initial_dir = '..'
    file_path = filedialog.askopenfilename(
        title='選擇實驗數據檔案 (runtime_data.npz)',
        initialdir=initial_dir,
        filetypes=[('NumPy檔案', '*.npz'), ('所有檔案', '*.*')]
    )
    root.destroy()
    if not file_path:
        print('未選擇檔案')
        return

    data = np.load(file_path, allow_pickle=True)
    error_list = data['error_list']
    Ts = float(data['Ts'])
    print(f'共 {len(error_list)} 步，每步 {len(error_list[0])} 點，Ts={Ts} 秒')

    # 時間段
    t_start_input = input("起始時間 (秒) [Enter=從頭]: ").strip()
    t_end_input = input("結束時間 (秒) [Enter=到尾]: ").strip()
    TIME_START = float(t_start_input) if t_start_input else None
    TIME_END = float(t_end_input) if t_end_input else None

    # 高通濾波
    hp_input = input("高通濾波頻率 Hz [Enter=不濾波]: ").strip()
    HIGHPASS_FREQ_HZ = float(hp_input) if hp_input else None

    # 異常值
    outlier_input = input("異常值門檻 (標準差倍數) [Enter=不移除]: ").strip()
    OUTLIER_THRESHOLD = float(outlier_input) if outlier_input else None

    # 合併誤差
    all_errors = np.concatenate(error_list)
    start_idx = int(TIME_START / Ts) if TIME_START else 0
    end_idx = int(TIME_END / Ts) if TIME_END else len(all_errors)
    start_idx = max(0, start_idx)
    end_idx = min(len(all_errors), end_idx)
    all_errors = all_errors[start_idx:end_idx]
    time_offset = start_idx * Ts
    all_errors = all_errors - np.mean(all_errors)

    # 異常值移除
    if OUTLIER_THRESHOLD:
        std = np.std(all_errors)
        outlier_mask = np.abs(all_errors) > OUTLIER_THRESHOLD * std
        if np.sum(outlier_mask) > 0:
            indices = np.arange(len(all_errors))
            valid_mask = ~outlier_mask
            all_errors[outlier_mask] = np.interp(indices[outlier_mask], indices[valid_mask], all_errors[valid_mask])
            print(f'移除異常值: {np.sum(outlier_mask)} 點 (>{OUTLIER_THRESHOLD}σ)')

    # 高通濾波
    if HIGHPASS_FREQ_HZ:
        nyquist = 0.5 / Ts
        b, a = scipy.signal.butter(4, HIGHPASS_FREQ_HZ / nyquist, btype='high')
        all_errors = scipy.signal.filtfilt(b, a, all_errors)
        print(f'高通濾波: {HIGHPASS_FREQ_HZ} Hz')

    # FFT
    N_fft = 10000
    window = np.hanning(len(all_errors))
    yf = scipy.fft.fft(window * all_errors, n=N_fft)
    xf = scipy.fft.fftfreq(N_fft, Ts)
    magnitude = np.abs(yf[:N_fft//2])
    omega = (2 * np.pi * xf)[:N_fft//2]

    freq_limit_rad = 500 * 2 * np.pi
    mask = omega < freq_limit_rad
    omega_plot = omega[mask]
    magnitude_plot = magnitude[mask]

    peaks, _ = scipy.signal.find_peaks(magnitude_plot, height=np.max(magnitude_plot) * 0.1, distance=10)
    sorted_idx = np.argsort(magnitude_plot[peaks])[::-1][:1]
    top_peaks = peaks[sorted_idx]
    top_freqs = omega_plot[top_peaks]
    top_mags = magnitude_plot[top_peaks]

    # 畫圖
    plt.figure(figsize=FIG_SIZE_WIDE)
    plt.subplot(2, 1, 1)
    time_axis = np.arange(len(all_errors)) * Ts + time_offset
    plt.plot(time_axis, all_errors, color='b')
    plt.xlabel('Time (s)')
    plt.ylabel('Error')
    title_parts = ['Full Error Time Domain']
    if TIME_START or TIME_END:
        title_parts.append(f'({time_offset:.1f}~{time_offset + len(all_errors)*Ts:.1f}s)')
    if HIGHPASS_FREQ_HZ:
        title_parts.append(f'HP {HIGHPASS_FREQ_HZ} Hz')
    plt.title(' '.join(title_parts))
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(omega_plot, magnitude_plot, color='b')
    plt.plot(top_freqs, top_mags, 'r*', markersize=12)
    plt.xlabel('Frequency (rad/s)')
    plt.ylabel('Magnitude')
    fft_title_parts = ['Full Error FFT Spectrum']
    if TIME_START or TIME_END:
        fft_title_parts.append(f'({time_offset:.1f}~{time_offset + len(all_errors)*Ts:.1f}s)')
    if HIGHPASS_FREQ_HZ:
        fft_title_parts.append(f'HP {HIGHPASS_FREQ_HZ} Hz')
    plt.title(' '.join(fft_title_parts))
    plt.xlim([0, freq_limit_rad])
    plt.ylim([0, np.max(magnitude_plot) * 1.2])
    for f, m in zip(top_freqs, top_mags):
        plt.text(f, m * 1.08, f'{f:.1f} rad/s ({f/(2*np.pi):.1f} Hz)',
                 ha='center', color='red', fontweight='bold')
    plt.grid(True)
    plt.tight_layout()

    output_path = os.path.join(os.path.dirname(file_path), 'error_full_fft.png')
    plt.savefig(output_path, dpi=150)
    print(f'圖片已儲存: {output_path}')
    for f, m in zip(top_freqs, top_mags):
        print(f'最高峰值: {f:.2f} rad/s ({f/(2*np.pi):.2f} Hz) - Magnitude: {m:.2f}')
    plt.show()


# ============================================================
# 主選單
# ============================================================

MENU = {
    '1': ('受控體波德圖繪製 (ID / Test / BUE / PRE)', plot_plant_ensemble),
    '2': ('路徑資料繪圖與匯出', plot_or_export_path),
    '3': ('隨機共振峰值上下界繪圖', plot_resonance_bounds),
    '4': ('動態FFT遮罩測試(產生動畫)', dynamic_fft_mask_animation),
    '5': ('載入實驗數據繪製開迴路波德圖', plot_experiment_openloop),
    '6': ('載入實驗數據繪製閉迴路極點圖', plot_experiment_OLoop_poles),
    '7': ('載入實驗數據機台共振頻譜分析', plot_resonance_spectrum),
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




