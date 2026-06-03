"""
實驗數據可視化與分析腳本
支持單個實驗數據文件或包含多個實驗數據的文件夾
- 選擇單個 .npz 文件：顯示該實驗的詳細信息和圖表
- 選擇文件夾：統計分析所有實驗數據並繪製均值和置信區間圖表
"""
import os
import sys
import io
import warnings
import glob
import tkinter as tk
from tkinter import filedialog

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import control as ctrl
import imageio.v2 as imageio
import scipy.fft
import scipy.signal

import CNC


# ============================================================
# 執行模式設定
# ============================================================
BATCH_MODE = True  # True = 批次處理（使用 BATCH_EXPERIMENTS）, False = 手動選擇

# 手動模式設定（BATCH_MODE = False 時使用）
SHOW_EVENT_LINES = True   # 設為 True 顯示所有事件線，False 則不顯示
RESONANCE_TIME = 3        # 共振檢測時間（秒），設為 None 則不顯示此線

# 批次處理配置（BATCH_MODE = True 時使用）
# 格式: {
#     'path': 資料夾路徑（相對於 ExperimentData 或絕對路徑）,
#     'mode': 'single' 或 'multi',
#     'show_events': True/False,
#     'resonance_time': 秒數或 None
# }
BATCH_EXPERIMENTS = [
    # === 單實驗 ===
    {'path': 'test1-1', 'mode': 'single', 'show_events': True, 'resonance_time': 3},
    {'path': 'test1-2', 'mode': 'single', 'show_events': False, 'resonance_time': None},
    {'path': 'test1-3', 'mode': 'single', 'show_events': False, 'resonance_time': None},
    {'path': 'test2-1', 'mode': 'single', 'show_events': True, 'resonance_time': 3},
    {'path': 'test3-1', 'mode': 'single', 'show_events': False, 'resonance_time': None},
    {'path': 'test4-1', 'mode': 'single', 'show_events': False, 'resonance_time': None},
    {'path': 'test4-2', 'mode': 'single', 'show_events': False, 'resonance_time': None},
    {'path': 'test5-1', 'mode': 'single', 'show_events': False, 'resonance_time': None},
    {'path': 'test6-1模', 'mode': 'single', 'show_events': False, 'resonance_time': None},
    {'path': 'test6-2模', 'mode': 'single', 'show_events': False, 'resonance_time': None},
    
    # === 雙模型模擬 ===
    {'path': '雙模型模擬/BUE_0~1Hz_有共振', 'mode': 'single', 'show_events': True, 'resonance_time': 0},
    {'path': '雙模型模擬/BUE_0~1Hz_無共振', 'mode': 'single', 'show_events': False, 'resonance_time': None},
    {'path': '雙模型模擬/PRE_0~1Hz_有共振', 'mode': 'single', 'show_events': True, 'resonance_time': 0},
    {'path': '雙模型模擬/PRE_0~1Hz_無共振', 'mode': 'single', 'show_events': False, 'resonance_time': None},
    
    # === 多實驗統計 ===
    {'path': 'test1多次模擬', 'mode': 'multi', 'show_events': False, 'resonance_time': None},
    {'path': 'test2多次模擬', 'mode': 'multi', 'show_events': False, 'resonance_time': None},
    {'path': 'test4多次模擬', 'mode': 'multi', 'show_events': False, 'resonance_time': None},
    {'path': 'test6-1多次模擬', 'mode': 'multi', 'show_events': False, 'resonance_time': None},
    {'path': 'test6-2多次模擬', 'mode': 'multi', 'show_events': False, 'resonance_time': None},
]

# ============================================================
# 繪圖參數
# ============================================================

# 圖片尺寸 (寬, 高) - 需被16整除以相容影片編碼
FIG_SIZE_SINGLE = (7.68, 5.76)    # 768x576
FIG_SIZE_WIDE = (11.52, 5.76)     # 1152x576
FIG_SIZE_MULTI = (10.24, 10.24)   # 1024x1024

# 字體大小（圖片越小，字體相對越大）
FONT_TITLE = 24    # 圖表標題
FONT_LABEL = 20    # 座標軸標籤
FONT_TICK = 18     # 刻度數字
FONT_LEGEND = 18   # 圖例

# 套用字體設定
matplotlib.rcParams.update({
    'axes.titlesize': FONT_TITLE,
    'axes.labelsize': FONT_LABEL,
    'xtick.labelsize': FONT_TICK,
    'ytick.labelsize': FONT_TICK,
    'legend.fontsize': FONT_LEGEND,
})


# ============================================================
# 繪圖工具類
# ============================================================

def FT_error(data, Ts):
    """傅立葉轉換（用於誤差 FFT 分析）"""
    pdl = len(data)
    hanning_window = np.hanning(pdl)
    windowed_data = hanning_window * data
    # 傅立葉轉換
    N = int(1/Ts)
    yf = scipy.fft.fft(windowed_data, N)
    freq_hz = scipy.fft.fftfreq(N, Ts)
    # 只保留正頻部分並轉換為 rad/s
    magnitude = np.abs(yf[:N // 2]) / pdl
    omega = 2 * np.pi * freq_hz[:N // 2]
    return omega, magnitude


def plot_error_fft_frame(step, ek, Ts, output, Wgc=None, folder='fft_frames'):
    """繪製單步的誤差 FFT 圖"""
    # 去除 DC 分量（只對 ek，與 find_resonance 一致）
    ek_processed = ek - ek[0]
    # output 不去 DC，與 CNC.find_resonance 保持一致
    
    # 傅立葉轉換
    eFTx, eFTy = FT_error(ek_processed, Ts)
    _, oFTy = FT_error(output, Ts)  # output 不處理
    
    # 找到峰值
    peaks_position, _ = scipy.signal.find_peaks(eFTy)
    
    # 繪圖
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIG_SIZE_SINGLE)
    
    # 上圖：時域誤差
    time_axis = np.arange(len(ek)) * Ts * 1000  # 轉換為 ms
    ax1.plot(time_axis, ek, 'b-', linewidth=1, label='Error')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Error (μm)')
    ax1.set_title(f'Step {step} - Time Domain Error', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # 下圖：頻域分析
    ax2.plot(eFTx, eFTy, 'b-', linewidth=1.5, label='Error FFT', alpha=0.8)
    ax2.plot(eFTx, oFTy, 'g--', linewidth=1, label='Output FFT', alpha=0.6)
    
    # 標記峰值
    if len(peaks_position) > 0:
        ax2.plot(eFTx[peaks_position], eFTy[peaks_position], 'rx', 
                markersize=6, alpha=0.5, label='All Peaks')
    
    # 標記 ω_gc 和閾值
    if Wgc is not None:
        ax2.axvline(x=Wgc, color='orange', linestyle='--', 
                   linewidth=2, alpha=0.7, label=f'ω_gc = {Wgc:.2f} rad/s')
        # 標記共振檢測閾值
        mean_output = np.mean(oFTy)
        threshold = mean_output * 5
        ax2.axhline(y=threshold, color='red', linestyle=':', 
                   linewidth=1.5, alpha=0.5, label=f'Threshold = {threshold:.4f}')
        
        # 標記通過檢測的共振峰值
        if len(peaks_position) > 0:
            mask = (eFTx[peaks_position] > Wgc) & (eFTy[peaks_position] > threshold)
            resonance_peaks = peaks_position[mask]
            if len(resonance_peaks) > 0:
                ax2.plot(eFTx[resonance_peaks], eFTy[resonance_peaks], 'r*', 
                        markersize=15, label='Detected Resonance', zorder=5)
                # 標註頻率
                for idx in resonance_peaks:
                        ax2.text(eFTx[idx], eFTy[idx]*1.1, f'{eFTx[idx]:.0f} rad/s', 
                            ha='center', color='red', fontweight='bold')
    
    ax2.set_xlabel('Frequency (rad/s)')
    ax2.set_ylabel('Magnitude')
    ax2.set_title(f'Step {step} - Frequency Domain Analysis', fontweight='bold')
    max_freq_rad = 250 * 2 * np.pi  # 對應原先 0-250 的頻段
    ax2.set_xlim(0, max_freq_rad)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    # 保存圖片
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = os.path.join(folder, f'fft_frame_{step:03d}.png')
    plt.savefig(filename, dpi=100)
    plt.close()
    
    return filename


def analyze_open_loop_margins(controller, plant, Ts, omega=None):
    """計算手動 GM/PM 與交越頻率，提供單實驗及多實驗共用"""
    OLoop = ctrl.minreal(ctrl.ss2tf(controller * plant), tol=1e-3, verbose=False)
    mag, phase, freq = ctrl.bode(OLoop, omega=omega, plot=False)
    mag = np.asarray(mag).reshape(-1)
    phase = np.asarray(phase).reshape(-1)
    freq = np.asarray(freq).reshape(-1)
    phase_deg_curve = np.rad2deg(phase)

    manual_wgc = None
    manual_pm = None
    if Ts and Ts > 0:
        try:
            manual_wgc = CNC.find_Wgc(controller, plant, Ts)
        except Exception as err:
            print(f"Failed to compute ω_gc: {err}")
            manual_wgc = None

    if manual_wgc is not None and manual_wgc > 0:
        manual_wgc = float(manual_wgc)
        try:
            _, phase_resp, _ = ctrl.freqresp(OLoop, [manual_wgc])
            phase_deg = float(np.rad2deg(np.asarray(phase_resp).reshape(-1)[0]))
            manual_pm = phase_deg + 180.0
        except Exception as err:
            print(f"Failed to compute manual PM: {err}")
            manual_pm = None
    else:
        manual_wgc = None

    manual_wpc = None
    manual_gm_db = None
    phase_shifted = phase_deg_curve + 180.0
    phase_shifted[np.isclose(phase_shifted, 0.0, atol=1e-3)] = 0.0
    crossings = []
    zero_hits = np.where(phase_shifted == 0.0)[0]
    crossings.extend(freq[zero_hits].tolist())
    products = phase_shifted[:-1] * phase_shifted[1:]
    sign_change_idx = np.where(products < 0)[0]
    for idx in sign_change_idx:
        p1, p2 = phase_shifted[idx], phase_shifted[idx + 1]
        w1, w2 = freq[idx], freq[idx + 1]
        if p2 != p1:
            ratio = -p1 / (p2 - p1)
            w_cross = w1 + ratio * (w2 - w1)
            if np.isfinite(w_cross):
                crossings.append(float(w_cross))
    crossings = [w for w in crossings if np.isfinite(w) and w > 0]
    if crossings:
        manual_wpc = max(crossings)
        try:
            mag_resp, _, _ = ctrl.freqresp(OLoop, [manual_wpc])
            mag_val = float(np.asarray(mag_resp).reshape(-1)[0])
            if mag_val > 0:
                manual_gm_db = -20.0 * np.log10(mag_val)
        except Exception as err:
            print(f"Failed to compute manual GM: {err}")
            manual_gm_db = None

    return OLoop, {
        'freq': freq,
        'mag_curve': mag,
        'phase_curve_deg': phase_deg_curve,
        'wgc': manual_wgc,
        'pm': manual_pm,
        'wpc': manual_wpc,
        'gm_db': manual_gm_db,
    }


class PlotExporter:
    """繪圖導出工具類"""
    def __init__(self, folder='frames', video_name='frequency_response.mp4', fps=5):
        self.folder = folder
        self.video_name = video_name
        self.fps = fps
        self.step = 0

        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        self.saved_frames = []

    def plot_frame(self, CC, plant, FC, manual_add_FC):
        """繪製單張 Bode 圖 + FC 點，並儲存圖片"""
        plt.figure(figsize=FIG_SIZE_SINGLE)
        OLoop = ctrl.minreal(ctrl.ss2tf(CC * plant), tol=1e-3, verbose=False)
        mag, _, oma = ctrl.bode(OLoop, dB=True, omega_limits=[1e-2, 3e3], plot=False)
        # 畫OLoop
        plt.plot(oma, 20 * np.log10(mag), color='b')
        # 畫手動添加FC點
        plt.scatter(manual_add_FC[:, 0], 20 * np.log10(manual_add_FC[:, 1]), color='g')
        # 畫Actor產生FC點
        plt.scatter(FC[:, 0], 20 * np.log10(FC[:, 1]), color='r')
        # 畫全部FC連線
        combined = np.vstack([FC, manual_add_FC])  # 神經網路跟手動FC合併
        new_FC = combined[np.argsort(combined[:, 0])]  # 排序
        plt.plot(new_FC[:, 0], 20 * np.log10(new_FC[:, 1]), color='r')

        plt.grid()
        plt.xscale('log')
        plt.xlim(1e-2, 1e4)
        plt.ylim(-70, 70)
        plt.xlabel("Frequency (rad/s)")
        plt.ylabel("Magnitude (dB)")
        plt.title(f'Step {self.step + 1}')

        filename = f'{self.folder}/frame_{self.step:03d}.png'
        plt.savefig(filename)
        plt.close('all')

        self.saved_frames.append(filename)
        self.step += 1

    def plot_error(self, error, Ts=0.001, events=None, total_length=None):
        """繪製誤差圖
        
        Parameters:
        -----------
        error : list
            誤差列表
        Ts : float
            採樣時間
        events : dict, optional
            事件時間點字典
        total_length : int, optional
            總繪圖長度（採樣點數），若 error 長度不足則補零
        """
        combined_data = np.concatenate(error)
        
        # 若指定 total_length 且 error 長度不足，則補零
        if total_length is not None and len(combined_data) < total_length:
            padding = np.zeros(total_length - len(combined_data))
            combined_data = np.concatenate([combined_data, padding])
        
        time = np.arange(len(combined_data)) * Ts
        
        plt.figure(figsize=FIG_SIZE_SINGLE)
        plt.plot(time, combined_data, linewidth=1.5)
        
        if not np.any(np.abs(combined_data) > 30):
            plt.ylim(-30, 30)
        
        # 添加事件標記線
        if events:
            event_styles = {
                'resonance_detected': {'color': 'red', 'linestyle': (0, (1, 1)), 'label': 'Add Resonance'},              # 紅色密集點線
                'switch_controller': {'color': 'blue', 'linestyle': (0, (5, 5)), 'label': 'Switch Model'},              # 藍色虛線
                'first_manual_fc': {'color': 'green', 'linestyle': (0, (3, 1, 1, 1)), 'label': 'RSS add FC'},      # 綠色點虛線
                'first_fc_controller_used': {'color': 'purple', 'linestyle': (0, (5, 1, 1, 1, 1, 1)), 'label': 'Controller Apply'}  # 紫色雙點虛線
            }
            
            event_order = ['resonance_detected', 'switch_controller', 'first_manual_fc', 'first_fc_controller_used']
            for event_name in event_order:
                if event_name in events and events[event_name] is not None:
                    style = event_styles[event_name]
                    plt.axvline(x=events[event_name], color=style['color'], linestyle=style['linestyle'], 
                               linewidth=3.0, alpha=0.8, label=style['label'])
        
        plt.title("Error")
        plt.xlabel("time(s)")
        plt.ylabel("Magnitude(um)")
        plt.grid()
        if events:
            plt.legend(loc='upper right')
        
        print("RMS Error: ", np.sqrt(np.mean(combined_data**2)))
        plt.show()

    def plot_margins(self, CC_list, plant, pdl, Ts=0.001, events=None):
        """
        繪製控制器性能指標隨時間變化圖
        
        Parameters:
        -----------
        CC_list : list
            每步的控制器列表，長度為總步數（例如50）
        plant : control.TransferFunction or control.StateSpace or dict
            受控對象，可以是傳遞函數或包含'v2p'鍵的字典
        pdl : int
            每步的採樣點數（每步時長 = pdl * Ts）
        Ts : float
            採樣時間，默認0.001秒
        events : dict, optional
            事件時間點字典，鍵為事件名稱，值為時間（秒）
            例如：{'switch_controller': 25.5, 'first_manual_fc': 10.2, 'resonance_detected': 5.0}
        """
        plant = ctrl.tf2ss(plant)
        num_steps = len(CC_list)
        time_per_step = pdl * Ts  # 每步的時間長度（秒）

        # 初始化數據存儲
        GM_list = []
        PM_list = []
        Wgc_list = []
        slope_list = []
        time_points = []

        for step, CC in enumerate(CC_list):
            OLoop, metrics = analyze_open_loop_margins(CC, plant, Ts)
            wgc = metrics['wgc']
            pm = metrics['pm']
            gm_db = metrics['gm_db']

            if wgc is not None and wgc > 0:
                delta_w = 0.01 * wgc if wgc > 0 else None
                if delta_w and delta_w > 0:
                    w1 = max(wgc - delta_w, wgc * 0.5, 1e-6)
                    w2 = wgc + delta_w
                    if w2 <= w1:
                        w2 = wgc * 1.01
                    mag_local, _, _ = ctrl.bode(OLoop, dB=True, omega=[w1, w2], plot=False)
                    mag_local = np.asarray(mag_local).reshape(-1)
                    slope = (20*np.log10(mag_local[1]) - 20*np.log10(mag_local[0])) / (np.log10(w2) - np.log10(w1))
                else:
                    slope = np.nan
            else:
                slope = np.nan

            GM_list.append(gm_db if gm_db is not None else np.nan)
            PM_list.append(pm if pm is not None else np.nan)
            Wgc_list.append(wgc if wgc is not None else np.nan)
            slope_list.append(slope)
            time_points.append(step * time_per_step)

        # 繪圖
        fig, axes = plt.subplots(4, 1, figsize=FIG_SIZE_MULTI)
         
        # GM 圖
        axes[0].plot(time_points, GM_list, 'b-', linewidth=2)
        axes[0].set_ylabel('Gain Margin (dB)')
        axes[0].grid(True)
        axes[0].set_title('Controller Performance Metrics over Time')
         
        # PM 圖
        axes[1].plot(time_points, PM_list, 'r-', linewidth=2)
        axes[1].set_ylabel('Phase Margin (deg)')
        axes[1].grid(True)
        
        # Wgc 圖
        axes[2].plot(time_points, Wgc_list, 'g-', linewidth=2)
        axes[2].set_ylabel('Wgc (rad/s)')
        axes[2].grid(True)
        
        # 斜率圖
        axes[3].plot(time_points, slope_list, 'm-', linewidth=2)
        axes[3].set_ylabel('Slope at Wgc (dB/dec)')
        axes[3].set_xlabel('Time (s)')
        axes[3].grid(True)
        
        # 添加事件標記線
        if events:
            event_styles = {
                'resonance_detected': {'color': 'red', 'linestyle': (0, (1, 1)), 'label': 'Add Resonance'},              # 紅色密集點線
                'switch_controller': {'color': 'blue', 'linestyle': (0, (5, 5)), 'label': 'Switch Model'},              # 藍色虛線
                'first_manual_fc': {'color': 'green', 'linestyle': (0, (3, 1, 1, 1)), 'label': 'RSS add FC'},      # 綠色點虛線
                'first_fc_controller_used': {'color': 'purple', 'linestyle': (0, (5, 1, 1, 1, 1, 1)), 'label': 'Controller Apply'}  # 紫色雙點虛線
            }
            
            # 按照指定順序添加事件線，確保圖例順序正確
            event_order = ['resonance_detected', 'switch_controller', 'first_manual_fc', 'first_fc_controller_used']
            
            for event_name in event_order:
                if event_name in events and events[event_name] is not None:
                    style = event_styles[event_name]
                    for ax in axes:
                        ax.axvline(x=events[event_name], color=style['color'], linestyle=style['linestyle'], 
                                   linewidth=3.0, alpha=0.8, label=style['label'])
            
            # 只在第一個子圖添加圖例
            axes[0].legend(loc='upper right')
         
        plt.tight_layout()
        plt.show()

    def save_mp4(self):
        """將所有儲存的圖片製作成 MP4"""
        with imageio.get_writer(self.video_name, fps=self.fps, codec='libx264', quality=8) as writer:
            for filename in self.saved_frames:
                image = imageio.imread(filename)
                writer.append_data(image)
        self.saved_frames = []
        self.step = 0


# ============================================================
# 數據處理函數
# ============================================================

def unwrap_tf(obj):
    """解包裝 numpy 儲存的 TransferFunction 對象"""
    if isinstance(obj, np.ndarray) and obj.ndim == 0:
        return obj.item()  # 從 0 維陣列提取對象
    return obj


def select_experiment_data():
    """彈出視窗選擇實驗數據檔案或文件夾
    返回: (path, is_folder) - 路徑和是否為文件夾的標誌
    """
    root = tk.Tk()
    root.withdraw()
    
    initial_dir = os.path.join("..", "ExperimentData")
    if not os.path.exists(initial_dir):
        initial_dir = ".."
    
    # 創建選擇對話框
    choice = tk.messagebox.askyesno(
        "選擇分析模式",
        "是否選擇文件夾進行多實驗統計分析？\n\n"
        "是 - 選擇文件夾（多實驗統計）\n"
        "否 - 選擇單個文件（單實驗詳細分析）"
    )
    
    if choice:
        # 選擇文件夾
        folder_path = filedialog.askdirectory(
            title="選擇包含多個實驗數據的文件夾",
            initialdir=initial_dir
        )
        root.destroy()
        return folder_path, True
    else:
        # 選擇文件
        file_path = filedialog.askopenfilename(
            title="選擇實驗數據檔案 (runtime_data.npz)",
            initialdir=initial_dir,
            filetypes=[("NumPy檔案", "*.npz"), ("所有檔案", "*.*")]
        )
        root.destroy()
        return file_path, False


def load_experiment_data(file_path):
    """讀取實驗數據（自動解包裝 numpy 儲存的對象）"""
    if not file_path:
        print("未選擇檔案")
        return None
    
    try:
        data = np.load(file_path, allow_pickle=True)
        # 解包裝所有可能被 numpy 包裝的對象（plant, CC 等）
        data_dict = dict(data)
        for key in data_dict:
            data_dict[key] = unwrap_tf(data_dict[key])
        print(f"✓ 成功載入實驗數據: {file_path}\n")
        return data_dict
    except Exception as e:
        print(f"✗ 載入失敗: {e}")
        return None


def load_all_experiments(folder_path):
    """加載文件夾內所有實驗數據"""
    # 查找所有 runtime_data.npz 或 simulation_data.npz 文件
    npz_files = glob.glob(os.path.join(folder_path, "**", "*data.npz"), recursive=True)
    
    if not npz_files:
        print(f"❌ 在 {folder_path} 中未找到任何 .npz 數據文件")
        return []
    
    print(f"✓ 找到 {len(npz_files)} 個數據文件")
    
    all_data = []
    for file_path in npz_files:
        try:
            data = np.load(file_path, allow_pickle=True)
            data_dict = dict(data)
            for key in data_dict:
                data_dict[key] = unwrap_tf(data_dict[key])
            all_data.append(data_dict)
            print(f"  ✓ 加載: {os.path.basename(os.path.dirname(file_path))}")
        except Exception as e:
            print(f"  ✗ 加載失敗 {os.path.basename(file_path)}: {e}")
    
    return all_data


def align_and_extract_errors(all_data):
    """對齊所有實驗的誤差數據（處理長度不同的情況）"""
    all_errors = []
    min_length = float('inf')
    
    # 找出最短的誤差序列長度
    for data in all_data:
        error_list = data['error_list']
        combined_error = np.concatenate(error_list)
        all_errors.append(combined_error)
        min_length = min(min_length, len(combined_error))
    
    # 截斷所有序列到相同長度
    aligned_errors = np.array([error[:min_length] for error in all_errors])
    
    return aligned_errors


def compute_statistics(aligned_errors):
    """計算統計量：均值、標準差、置信區間"""
    mean_error = np.mean(aligned_errors, axis=0)
    std_error = np.std(aligned_errors, axis=0)
    
    # 計算95%置信區間 (1.96 * std / sqrt(n))
    n_experiments = aligned_errors.shape[0]
    confidence_interval = 1.96 * std_error / np.sqrt(n_experiments)
    
    return mean_error, std_error, confidence_interval


def align_and_extract_margins(all_data):
    """對齊所有實驗的性能指標數據（GM, PM, Wgc, Slope）"""
    all_GM = []
    all_PM = []
    all_Wgc = []
    all_Slope = []
    min_length = float('inf')
    
    # 提取數據
    for data in all_data:
        CC_list = data['CC_list']
        pdl = data['pdl']
        Ts = data['Ts']
        plant = ctrl.tf2ss(data['ID_Plant_v2p'])
        
        GM_list = []
        PM_list = []
        Wgc_list = []
        Slope_list = []
        
        # 計算每步的性能指標
        for CC in CC_list:
            OLoop, metrics = analyze_open_loop_margins(CC, plant, Ts)
            wgc = metrics['wgc']
            pm = metrics['pm']
            gm_db = metrics['gm_db']

            if wgc is not None and wgc > 0:
                delta_w = 0.01 * wgc
                if delta_w > 0:
                    w1 = max(wgc - delta_w, wgc * 0.5, 1e-6)
                    w2 = wgc + delta_w
                    if w2 <= w1:
                        w2 = wgc * 1.01
                    mag_local, _, _ = ctrl.bode(OLoop, dB=True, omega=[w1, w2], plot=False)
                    mag_local = np.asarray(mag_local).reshape(-1)
                    slope = (20*np.log10(mag_local[1]) - 20*np.log10(mag_local[0])) / (np.log10(w2) - np.log10(w1))
                else:
                    slope = np.nan
            else:
                slope = np.nan

            GM_list.append(gm_db if gm_db is not None else np.nan)
            PM_list.append(pm if pm is not None else np.nan)
            Wgc_list.append(wgc if wgc is not None else np.nan)
            Slope_list.append(slope)
        
        all_GM.append(np.array(GM_list))
        all_PM.append(np.array(PM_list))
        all_Wgc.append(np.array(Wgc_list))
        all_Slope.append(np.array(Slope_list))
        min_length = min(min_length, len(GM_list))
    
    # 截斷到相同長度
    aligned_GM = np.array([gm[:min_length] for gm in all_GM])
    aligned_PM = np.array([pm[:min_length] for pm in all_PM])
    aligned_Wgc = np.array([wgc[:min_length] for wgc in all_Wgc])
    aligned_Slope = np.array([slope[:min_length] for slope in all_Slope])
    
    return aligned_GM, aligned_PM, aligned_Wgc, aligned_Slope


def display_basic_info(data, return_string=False):
    """顯示基本實驗資訊
    
    Parameters:
    -----------
    data : dict
        實驗數據
    return_string : bool
        是否返回字符串而不是打印
    """
    lines = []
    lines.append("=" * 60)
    lines.append("實驗基本資訊")
    lines.append("=" * 60)
    
    lines.append("\n【執行資訊】")
    if 'execution_script' in data:
        lines.append(f"  執行程式: {data['execution_script']}")
    
    lines.append("\n【時間資訊】")
    lines.append(f"  實驗時間: {data['experiment_datetime']}")
    lines.append(f"  實驗時長: {int(data['experiment_duration']//60)}分{int(data['experiment_duration']%60)}秒")
    lines.append(f"  實際步數: {data['actual_steps']}")
    
    lines.append("\n【Model資訊】")
    use_switch = data.get('use_switch_model', False)
    lines.append(f"  使用模式: {'雙模型切換' if use_switch else '單模型'}")
    lines.append(f"  Model1 檔案: {data['model1_filename']}")
    lines.append(f"  Model1 輪數: {data['model1_iteration']}")
    if use_switch and data.get('model2_filename') is not None:
        lines.append(f"  Model2 檔案: {data['model2_filename']}")
        lines.append(f"  Model2 輪數: {data['model2_iteration']}")
        if data.get('switch_step') is not None:
            lines.append(f"  切換步數: {data['switch_step']}")
    
    lines.append("\n【系統參數】")
    lines.append(f"  採樣時間 Ts: {data['Ts']} s")
    lines.append(f"  路徑區間長度 pdl: {data['pdl']} ms")
    lines.append(f"  頻率限制點數量 numFC: {data['numFC']}")
    fft_limit_freq_rad = data['fft_limit_freq'] * 2 * np.pi
    lines.append(f"  FFT頻率上限: {fft_limit_freq_rad:.2f} rad/s")
    lines.append(f"  低頻限制點數: {data['num_low_freq_FC']}")
    lines.append(f"  X軸極點增益: {data['x_polegain']}")
    if 'z_polegain' in data:
        lines.append(f"  Z軸極點增益: {data['z_polegain']}")
    
    lines.append("\n【CNC參數】")
    cnc = data['CNC_params']
    lines.append(f"  Lq: {cnc['Lq']}")
    
    lines.append("\n【參考路徑資訊】")
    if 'path_length' in data:
        lines.append(f"  路徑總長度: {data['path_length']}")
    elif 'reference_path' in data:
        lines.append(f"  路徑總長度: {len(data['reference_path'])}")
    lines.append(f"  路徑區間數: {data['num_districts']}")
    path_len = data.get('path_length', len(data.get('reference_path', [])))
    lines.append(f"  路徑時間: {path_len * data['Ts']} s")
    
    lines.append("\n【誤差統計】")
    error_list = data['error_list']
    combined_error = np.concatenate(error_list)
    rms_error = np.sqrt(np.mean(combined_error**2))
    lines.append(f"  RMS Error: {rms_error:.4f} μm")
    lines.append(f"  Max Error: {np.max(np.abs(combined_error)):.4f} μm")
    lines.append(f"  Mean Error: {np.mean(np.abs(combined_error)):.4f} μm")
    
    text = "\n".join(lines)
    if return_string:
        return text
    else:
        print(text)


def display_step_details(data, return_string=False):
    """顯示每步詳細狀況
    
    Parameters:
    -----------
    data : dict
        實驗數據
    return_string : bool
        是否返回字符串而不是打印
    """
    lines = []
    lines.append("=" * 60)
    lines.append("每步詳細狀況")
    lines.append("=" * 60)
    lines.append(f"{'Step':<6} | {'Status':<15} | {'Freq (rad/s)':<20} | {'Mag (dB)':<12} | {'Model':<15}")
    lines.append("-" * 60)
    
    status_list = data['status_list']
    resonance_freq_list = data['resonance_freq_list']
    resonance_gain_list = data['resonance_gain_list']
    model_used = data.get('model_used', [])
    
    for step in range(len(status_list)):
        status = status_list[step]
        freq_rad_per_s = resonance_freq_list[step]  # 單位是 rad/s
        mag_dB = resonance_gain_list[step]  # 已经是 dB 值
        model = model_used[step] if step < len(model_used) else 'N/A'
        
        freq_display = f"{freq_rad_per_s:.2f}" if freq_rad_per_s > 0 else "0"
        
        lines.append(
            f"{step:<6} | "
            f"{status:<15} | "
            f"{freq_display:<20} | "
            f"{mag_dB:<12.5g} | "
            f"{model:<15}"
        )
    
    lines.append("\n" + "=" * 60 + "\n")
    
    text = "\n".join(lines)
    if return_string:
        return text
    else:
        print(text)


def plot_reference_path(data, experiment_folder):
    """繪製參考路徑並保存"""
    path = data['reference_path']
    Ts = data['Ts']
    t = np.arange(0, len(path) * Ts, Ts)
    
    inputdata_plot = np.column_stack((t, path))
    plt.figure(figsize=FIG_SIZE_WIDE)
    plt.plot(inputdata_plot[:, 0], inputdata_plot[:, 1])
    plt.title('Path')
    plt.xlabel('Time (s)')
    plt.ylabel('Magnitude(mm)')
    plt.grid(True)
    
    save_path = os.path.join(experiment_folder, 'reference_path.png')
    plt.savefig(save_path)
    plt.close()
    print(f"參考路徑圖已保存: {save_path}")


def plot_error(data, plotter, experiment_folder, show_events=True, resonance_time=None):
    """繪製誤差圖並保存
    
    Parameters:
    -----------
    data : dict
        實驗數據
    plotter : PlotExporter
        繪圖導出工具
    experiment_folder : str
        實驗文件夾路徑
    show_events : bool
        是否顯示事件標記線
    resonance_time : float, optional
        共振檢測時間
    """
    error_list = data['error_list']
    pdl = data['pdl']
    Ts = data['Ts']
    
    # 使用參考路徑長度決定繪圖範圍（而非 error 長度）
    if 'reference_path' in data:
        total_length = len(data['reference_path'])
    elif 'path_length' in data:
        total_length = int(data['path_length'])
    else:
        total_length = None  # fallback to error length
    
    # 計算事件時間
    events = None
    if show_events:
        events = {}
        time_per_step = pdl * Ts
        
        # 1. 切換控制器時間
        if data.get('switch_step') is not None:
            events['switch_controller'] = data['switch_step'] * time_per_step
        
        # 2. 第一個 manual FC 被加入的時間
        manual_FC_list = data.get('manual_FC_list', [])
        first_fc_step = None
        for step, manual_fc in enumerate(manual_FC_list):
            if manual_fc is not None and len(manual_fc) > 0:
                events['first_manual_fc'] = step * time_per_step
                first_fc_step = step
                break
        
        # 2.1. 第一個限制點控制器被使用的時間（加入後的下一步）
        if first_fc_step is not None:
            events['first_fc_controller_used'] = (first_fc_step + 1) * time_per_step
        
        # 3. 共振檢測時間（由用戶指定）
        if resonance_time is not None:
            events['resonance_detected'] = resonance_time
    
    # 临时屏蔽print输出和plt.show()
    original_backend = matplotlib.get_backend()
    matplotlib.use('Agg')
    
    # 重定向stdout以屏蔽print，并屏蔽warnings
    original_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plotter.plot_error(error_list, Ts=Ts, events=events if events else None, total_length=total_length)
        
        save_path = os.path.join(experiment_folder, 'error.png')
        plt.savefig(save_path)
    finally:
        plt.close('all')  # 在切換backend前關閉所有圖形
        sys.stdout = original_stdout
        matplotlib.use(original_backend)
    
    print(f"誤差圖已保存: {save_path}")


def plot_margins(data, plotter, experiment_folder, show_events=True, resonance_time=None):
    """繪製控制器性能指標圖並保存
    
    Parameters:
    -----------
    show_events : bool, optional
        是否顯示事件標記線，默認為 True
    resonance_time : float, optional
        共振被檢測到的時間（秒），如果不提供則不顯示此標記
    """
    import control as ctrl
    
    CC_list = data['CC_list']
    pdl = data['pdl']
    Ts = data['Ts']
    plant = data['ID_Plant_v2p']
    
    # 計算事件時間
    events = None
    if show_events:
        events = {}
        time_per_step = pdl * Ts
        
        # 1. 切換控制器時間
        if data.get('switch_step') is not None:
            events['switch_controller'] = data['switch_step'] * time_per_step
        
        # 2. 第一個 manual FC 被加入的時間
        manual_FC_list = data.get('manual_FC_list', [])
        first_fc_step = None
        for step, manual_fc in enumerate(manual_FC_list):
            if manual_fc is not None and len(manual_fc) > 0:
                events['first_manual_fc'] = step * time_per_step
                first_fc_step = step
                break
        
        # 2.1. 第一個限制點控制器被使用的時間（加入後的下一步）
        if first_fc_step is not None:
            events['first_fc_controller_used'] = (first_fc_step + 1) * time_per_step
        
        # 3. 共振檢測時間（由用戶指定）
        if resonance_time is not None:
            events['resonance_detected'] = resonance_time
    
    # 临时屏蔽print输出和plt.show()
    original_backend = matplotlib.get_backend()
    matplotlib.use('Agg')
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plotter.plot_margins(CC_list, plant, pdl, Ts, events=events if events else None)
        
        save_path = os.path.join(experiment_folder, 'controller_margins.png')
        plt.savefig(save_path)
    finally:
        plt.close('all')
        # sys.stdout = original_stdout
        matplotlib.use(original_backend)
    
    print(f"控制器性能指標圖已保存: {save_path}")


def plot_frequency_response_mp4(data, plotter):
    """繪製每步的頻率響應並生成MP4"""
    import control as ctrl
    
    CC_list = data['CC_list']
    FC_list = data['FC_list']
    manual_FC_list = data['manual_FC_list']
    plant = data['ID_Plant_v2p']
    
    if isinstance(plant, ctrl.TransferFunction):
        plant = ctrl.tf2ss(plant)
    
    print(f"\n【生成頻率響應MP4】")
    print(f"  總步數: {len(CC_list)}")
    print(f"  正在生成圖片...")
    
    for i in range(len(CC_list)):
        CC = CC_list[i]
        FC = FC_list[i]
        manual_FC = manual_FC_list[i] if manual_FC_list[i].size > 0 else np.empty((0, 2))
        plotter.plot_frame(CC, plant, FC, manual_FC)
    
    print(f"  正在創建 MP4...")
    plotter.save_mp4()
    print(f"  MP4已生成完成！")


def plot_error_fft_mp4(data, experiment_folder):
    """繪製每步的誤差 FFT 並生成 MP4"""
    print(f"\n【生成誤差FFT MP4】")
    print(f"  總步數: {data['actual_steps']}")
    print(f"  正在生成圖片...")
    
    Ts = float(data["Ts"])
    pdl = int(data["pdl"])
    error_list = data['error_list']
    CC_list = data['CC_list']
    path = data['reference_path']
    plant = data['ID_Plant_v2p']
    
    # 準備輸出資料夾
    fft_folder = os.path.join(experiment_folder, 'fft_frames')
    
    # 清理舊的 frames
    if os.path.exists(fft_folder):
        for file in os.listdir(fft_folder):
            os.remove(os.path.join(fft_folder, file))
    
    saved_frames = []
    
    # 為每一步生成 FFT 圖
    for step in range(len(error_list)):
        ek = error_list[step]
        
        # 計算 output
        path_start = step * pdl
        path_end = (step + 1) * pdl
        path_district = path[path_start:path_end]
        output = path_district - ek
        
        # 計算 ω_gc（如果有 CC）
        Wgc = None
        if step < len(CC_list):
            try:
                CC = CC_list[step]
                Wgc = CNC.find_Wgc(CC, plant, Ts)
            except:
                pass
        
        # 繪製並保存
        filename = plot_error_fft_frame(step, ek, Ts, output, Wgc, fft_folder)
        saved_frames.append(filename)
    
    # 創建 MP4
    mp4_name = os.path.join(experiment_folder, 'error_fft.mp4')
    print(f'  正在創建 MP4...')
    
    with imageio.get_writer(mp4_name, fps=5, codec='libx264', quality=8) as writer:
        for filename in saved_frames:
            image = imageio.imread(filename)
            writer.append_data(image)
    
    print(f'  MP4已生成完成！\n')
    return mp4_name


# ============================================================
# 多實驗統計分析函數
# ============================================================

def _plot_statistics_subplot(ax, time, aligned_data, ylabel, title=None, show_individual=False):
    """繪製統計圖的通用子圖函數：均值 + 標準差"""
    mean_data = np.mean(aligned_data, axis=0)
    std_data = np.std(aligned_data, axis=0)
    
    # 显示单次实验曲线（可选）
    if show_individual:
        for i, data in enumerate(aligned_data):
            ax.plot(time, data, alpha=0.15, linewidth=0.5, color='gray',
                   label='Individual runs' if i == 0 else '')
    
    # 标准差阴影区域（使用浅蓝色，实心，先画）
    ax.fill_between(time, mean_data - std_data, mean_data + std_data,
                    alpha=1.0, color='skyblue', label='Mean ± 1σ', zorder=2)
    # 均值曲线（画在标准差上方）
    ax.plot(time, mean_data, color='darkblue', linewidth=2, label='Mean', zorder=3)
    
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    if title:
        ax.set_title(title)


def plot_error_statistics(aligned_errors, mean_error, std_error, 
                          Ts=0.001, save_path=None, show_individual=False):
    """繪製誤差統計圖"""
    time = np.arange(len(mean_error)) * Ts
    
    plt.figure(figsize=FIG_SIZE_SINGLE)
    
    if show_individual:
        for i, error in enumerate(aligned_errors):
            plt.plot(time, error, alpha=0.15, linewidth=0.5, color='gray', 
                    label='Individual runs' if i == 0 else '')
    
    # 标准差阴影区域（使用浅蓝色，实心，先画）
    plt.fill_between(time, mean_error - std_error, mean_error + std_error,
                     alpha=1.0, color='skyblue', label='Mean ± 1σ', zorder=2)
    # 均值曲线（画在标准差上方）
    plt.plot(time, mean_error, color='darkblue', linewidth=2, label='Mean', zorder=3)
    
    max_abs_error = np.max(np.abs(mean_error + std_error))
    if max_abs_error < 30:
        plt.ylim(-30, 30)
    
    plt.xlabel('time(s)')
    plt.ylabel('Magnitude(um)')
    plt.title('Error Statistics')
    plt.grid()
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"✓ 誤差統計圖已保存: {save_path}")
    plt.close()


def plot_margins_statistics(aligned_GM, aligned_PM, aligned_Wgc, aligned_Slope,
                           pdl, Ts=0.001, save_path=None, show_individual=False):
    """繪製控制器性能指標統計圖"""
    n_steps = aligned_GM.shape[1]
    time_points = np.arange(n_steps) * (pdl * Ts)
    
    fig, axes = plt.subplots(4, 1, figsize=FIG_SIZE_MULTI)
    
    # 使用通用函數繪製4個子圖
    _plot_statistics_subplot(axes[0], time_points, aligned_GM, 'Gain Margin (dB)',
                             f'Controller Performance Metrics Statistics (n={aligned_GM.shape[0]} runs)',
                             show_individual)
    _plot_statistics_subplot(axes[1], time_points, aligned_PM, 'Phase Margin (deg)',
                             None, show_individual)
    _plot_statistics_subplot(axes[2], time_points, aligned_Wgc, 'Wgc (rad/s)',
                             None, show_individual)
    _plot_statistics_subplot(axes[3], time_points, aligned_Slope, 'Slope at Wgc (dB/dec)',
                             None, show_individual)
    
    axes[3].set_xlabel('Time (s)')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"✓ 性能指標統計圖已保存: {save_path}")
    plt.close()


def print_statistics_summary(all_data, aligned_errors, mean_error, std_error, return_string=False):
    """打印統計摘要
    
    Parameters:
    -----------
    return_string : bool
        是否返回字符串而不是打印
    """
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("多實驗統計摘要")
    lines.append("=" * 70)
    
    lines.append(f"\n【實驗數量】: {len(all_data)} 次")
    lines.append(f"【數據點數】: {len(mean_error)} 個採樣點")
    
    # 共同參數（從第一筆數據取）
    first_data = all_data[0]
    lines.append(f"\n【共同參數】")
    lines.append(f"  採樣時間 Ts: {first_data['Ts']} s")
    lines.append(f"  路徑區間長度 pdl: {first_data['pdl']} ms")
    if 'model1_filename' in first_data:
        lines.append(f"  Model: {first_data['model1_filename']}")
    
    lines.append("\n【RMS誤差統計】")
    rms_list = []
    for data in all_data:
        error_list = data['error_list']
        combined_error = np.concatenate(error_list)
        rms = np.sqrt(np.mean(combined_error**2))
        rms_list.append(rms)
    
    lines.append(f"  平均 RMS: {np.mean(rms_list):.4f} μm")
    lines.append(f"  最小 RMS: {np.min(rms_list):.4f} μm")
    lines.append(f"  最大 RMS: {np.max(rms_list):.4f} μm")
    lines.append(f"  標準差:   {np.std(rms_list):.4f} μm")
    
    lines.append("\n【整體誤差統計】")
    all_errors_flat = aligned_errors.flatten()
    lines.append(f"  均值誤差: {np.mean(all_errors_flat):.4f} μm")
    lines.append(f"  標準差:   {np.std(all_errors_flat):.4f} μm")
    lines.append(f"  最大誤差: {np.max(np.abs(all_errors_flat)):.4f} μm")
    
    lines.append("\n【時間平均標準差】")
    lines.append(f"  平均標準差: {np.mean(std_error):.4f} μm")
    lines.append(f"  最大標準差: {np.max(std_error):.4f} μm")
    lines.append(f"  最小標準差: {np.min(std_error):.4f} μm")
    
    # 每次實驗的 RMS 列表
    lines.append("\n【各實驗 RMS】")
    for i, (data, rms) in enumerate(zip(all_data, rms_list)):
        # 嘗試取得實驗時間或資料夾名
        exp_time = data.get('experiment_datetime', f'實驗{i+1}')
        lines.append(f"  {exp_time}: {rms:.4f} μm")
    
    lines.append("\n" + "=" * 70 + "\n")
    
    text = "\n".join(lines)
    if return_string:
        return text
    else:
        print(text)


def save_experiment_info(data, experiment_folder):
    """保存實驗資訊到 txt 文件"""
    basic_info = display_basic_info(data, return_string=True)
    step_details = display_step_details(data, return_string=True)
    
    full_text = basic_info + "\n" + step_details
    
    save_path = os.path.join(experiment_folder, 'experiment_info.txt')
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(full_text)
    
    print(f"實驗資訊已保存: {save_path}")
    return save_path


def process_single_experiment(file_path):
    """處理單個實驗數據"""
    data = load_experiment_data(file_path)
    
    if data is None:
        return
    
    experiment_folder = os.path.dirname(file_path)
    frames_folder = os.path.join(experiment_folder, 'frames')
    video_path = os.path.join(experiment_folder, 'frequency_response.mp4')
    
    plotter = PlotExporter(folder=frames_folder, video_name=video_path)
    
    # 顯示並保存實驗資訊
    display_basic_info(data)
    display_step_details(data)
    save_experiment_info(data, experiment_folder)
    
    plot_reference_path(data, experiment_folder)
    
    plot_error(data, plotter, experiment_folder, show_events=SHOW_EVENT_LINES, resonance_time=RESONANCE_TIME)
    
    plot_margins(data, plotter, experiment_folder, show_events=SHOW_EVENT_LINES, resonance_time=RESONANCE_TIME)
    
    plot_frequency_response_mp4(data, plotter)
    
    plot_error_fft_mp4(data, experiment_folder)


def process_multiple_experiments(folder_path):
    """處理多個實驗數據（統計分析）"""
    print("="*70)
    print("多實驗數據統計分析")
    print("="*70)
    print(f"\n選擇的文件夾: {folder_path}\n")
    
    # 1. 加載所有實驗數據
    all_data = load_all_experiments(folder_path)
    if not all_data:
        return
    
    # 2. 對齊誤差數據
    print("\n正在對齊誤差數據...")
    aligned_errors = align_and_extract_errors(all_data)
    print(f"✓ 對齊完成: {aligned_errors.shape[0]} 次實驗, 每次 {aligned_errors.shape[1]} 個採樣點")
    
    # 3. 計算統計量
    print("\n正在計算統計量...")
    mean_error, std_error, _ = compute_statistics(aligned_errors)
    print("✓ 統計量計算完成")
    
    # 4. 打印統計摘要
    print_statistics_summary(all_data, aligned_errors, mean_error, std_error)
    
    # 4.1 保存統計摘要到 txt
    summary_text = print_statistics_summary(all_data, aligned_errors, mean_error, std_error, return_string=True)
    summary_path = os.path.join(folder_path, 'statistics_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print(f"✓ 統計摘要已保存: {summary_path}")
    
    # 5. 對齊性能指標數據
    print("\n正在提取和對齊性能指標數據...")
    try:
        aligned_GM, aligned_PM, aligned_Wgc, aligned_Slope = align_and_extract_margins(all_data)
        print(f"✓ 性能指標對齊完成: {aligned_GM.shape[0]} 次實驗, 每次 {aligned_GM.shape[1]} 步")
        has_margins = True
    except Exception as e:
        print(f"⚠ 性能指標提取失敗: {e}")
        print("  將跳過性能指標統計圖")
        has_margins = False
    
    # 6. 繪圖
    Ts = all_data[0]['Ts']
    pdl = all_data[0]['pdl']
    output_folder = folder_path
    
    print("\n正在生成圖表...")
    
    # 圖1: 誤差統計圖（均值 + 標準差）
    plot_error_statistics(
        aligned_errors, mean_error, std_error,
        Ts=Ts,
        save_path=os.path.join(output_folder, 'error_statistics.png'),
        show_individual=False  # 設為True可顯示每次實驗的曲線
    )
    
    # 圖2: 性能指標統計圖
    if has_margins:
        plot_margins_statistics(
            aligned_GM, aligned_PM, aligned_Wgc, aligned_Slope,
            pdl=pdl, Ts=Ts,
            save_path=os.path.join(output_folder, 'margins_statistics.png'),
            show_individual=False  # 設為True可顯示每次實驗的曲線
        )
    
    print("\n✓ 所有圖表生成完成！")
    print(f"✓ 圖表已保存到: {output_folder}")


# ============================================================
# 批次處理函數
# ============================================================

def get_full_path(path):
    """將相對路徑轉換為完整路徑"""
    if os.path.isabs(path):
        return path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    exp_data_dir = os.path.join(base_dir, '..', 'ExperimentData')
    return os.path.normpath(os.path.join(exp_data_dir, path))


def process_batch_experiment(config):
    """處理單個批次配置"""
    global SHOW_EVENT_LINES, RESONANCE_TIME
    
    path = get_full_path(config['path'])
    mode = config['mode']
    
    print("\n" + "=" * 70)
    print(f"處理: {config['path']}")
    print(f"模式: {'單實驗' if mode == 'single' else '多實驗'}")
    print(f"事件線: {config.get('show_events', True)}, 共振時間: {config.get('resonance_time', None)}")
    print("=" * 70)
    
    if not os.path.exists(path):
        print(f"❌ 路徑不存在: {path}")
        return False
    
    # 暫存並修改全局設定
    original_show = SHOW_EVENT_LINES
    original_res = RESONANCE_TIME
    SHOW_EVENT_LINES = config.get('show_events', True)
    RESONANCE_TIME = config.get('resonance_time', None)
    
    try:
        if mode == 'single':
            npz_file = os.path.join(path, 'runtime_data.npz')
            if not os.path.exists(npz_file):
                npz_file = os.path.join(path, 'simulation_data.npz')
            if not os.path.exists(npz_file):
                print(f"❌ 找不到數據文件: {path}")
                return False
            process_single_experiment(npz_file)
        elif mode == 'multi':
            process_multiple_experiments(path)
        else:
            print(f"❌ 未知模式: {mode}")
            return False
        print(f"✓ 完成: {config['path']}")
        return True
    except Exception as e:
        print(f"❌ 處理失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        SHOW_EVENT_LINES = original_show
        RESONANCE_TIME = original_res


def run_batch():
    """執行批次處理"""
    if not BATCH_EXPERIMENTS:
        print("\n⚠ BATCH_EXPERIMENTS 列表為空，請先在腳本中配置實驗資料夾。")
        return
    
    print("=" * 70)
    print("批次實驗數據可視化")
    print("=" * 70)
    print(f"\n共 {len(BATCH_EXPERIMENTS)} 個實驗待處理\n")
    
    success, fail = 0, 0
    for i, config in enumerate(BATCH_EXPERIMENTS, 1):
        print(f"\n[{i}/{len(BATCH_EXPERIMENTS)}] ", end="")
        if process_batch_experiment(config):
            success += 1
        else:
            fail += 1
    
    print("\n" + "=" * 70)
    print(f"批次處理完成 - 成功: {success}, 失敗: {fail}")
    print("=" * 70)


def main():
    # 根據 BATCH_MODE 決定執行模式
    if BATCH_MODE:
        run_batch()
    else:
        path, is_folder = select_experiment_data()
        if not path:
            print("未選擇檔案或文件夾")
            return
        if is_folder:
            process_multiple_experiments(path)
        else:
            process_single_experiment(path)


if __name__ == "__main__":
    main()
