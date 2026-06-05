import os
import tkinter as tk
from tkinter import filedialog

import numpy as np
import matplotlib.pyplot as plt
import scipy.fft
import scipy.signal

# ============================================================
# 配置參數
# ============================================================
HIGHPASS_FREQ_HZ = 50  # 高通濾波頻率 (Hz)，設為 None 則不濾波
TIME_START = None      # 起始時間 (秒)，設為 None 則從頭開始
TIME_END = None        # 結束時間 (秒)，設為 None 則到最後
OUTLIER_THRESHOLD = 1  # 異常值門檻（標準差倍數），設為 None 則不移除
# ============================================================


def select_data_file() -> str:
    """選擇 runtime_data.npz 檔案"""
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
    return file_path


def main():
    file_path = select_data_file()
    if not file_path:
        print('未選擇檔案，結束程式。')
        return

    # 載入數據
    data = np.load(file_path, allow_pickle=True)
    error_list = data['error_list']
    Ts = float(data['Ts'])

    print(f'共 {len(error_list)} 步')
    print(f'每步 {len(error_list[0])} 點')
    print(f'總共 {len(error_list) * len(error_list[0])} 點')
    print(f'採樣時間 Ts = {Ts} 秒\n')

    # 合併所有誤差
    all_errors = np.concatenate(error_list)
    total_time = len(all_errors) * Ts
    print(f'總時間: {total_time:.2f} 秒')
    
    # 時間段截取
    start_idx = int(TIME_START / Ts) if TIME_START else 0
    end_idx = int(TIME_END / Ts) if TIME_END else len(all_errors)
    start_idx = max(0, start_idx)
    end_idx = min(len(all_errors), end_idx)
    all_errors = all_errors[start_idx:end_idx]
    time_offset = start_idx * Ts
    
    if TIME_START or TIME_END:
        actual_start = start_idx * Ts
        actual_end = end_idx * Ts
        print(f'選取時間段: {actual_start:.2f} ~ {actual_end:.2f} 秒 ({len(all_errors)} 點)')
    
    all_errors = all_errors - np.mean(all_errors)

    # 移除異常值（脈衝尖峰）
    if OUTLIER_THRESHOLD:
        std = np.std(all_errors)
        threshold = OUTLIER_THRESHOLD * std
        outlier_mask = np.abs(all_errors) > threshold
        outlier_count = np.sum(outlier_mask)
        if outlier_count > 0:
            # 用線性插值替代異常值
            all_errors_clean = all_errors.copy()
            indices = np.arange(len(all_errors))
            valid_mask = ~outlier_mask
            all_errors_clean[outlier_mask] = np.interp(
                indices[outlier_mask], 
                indices[valid_mask], 
                all_errors[valid_mask]
            )
            print(f'移除異常值: {outlier_count} 點 (>{OUTLIER_THRESHOLD}σ, σ={std:.2f})')
            all_errors = all_errors_clean

    # 高通濾波 (時域處理)
    if HIGHPASS_FREQ_HZ:
        nyquist = 0.5 / Ts
        highpass_normalized = HIGHPASS_FREQ_HZ / nyquist
        b, a = scipy.signal.butter(4, highpass_normalized, btype='high')
        all_errors_filtered = scipy.signal.filtfilt(b, a, all_errors)
        print(f'高通濾波: {HIGHPASS_FREQ_HZ} Hz (Butterworth 4th order)')
    else:
        all_errors_filtered = all_errors

    # FFT with zero-padding for better frequency resolution
    N_original = len(all_errors_filtered)
    N_fft = 10000  # 增加FFT點數 (zero-padding)
    window = np.hanning(N_original)
    windowed_errors = window * all_errors_filtered
    yf = scipy.fft.fft(windowed_errors, n=N_fft)
    xf = scipy.fft.fftfreq(N_fft, Ts)

    magnitude = np.abs(yf[:N_fft//2])
    omega = (2 * np.pi * xf)[:N_fft//2]

    freq_resolution_rad = (1 / Ts) / N_fft * 2 * np.pi
    print(f'FFT點數: {N_fft}')
    print(f'頻率分辨率: {freq_resolution_rad:.6f} rad/s')
    freq_limit_rad = 500 * 2 * np.pi
    print(f'0-{freq_limit_rad:.0f} rad/s 數據點數: {len(omega[omega < freq_limit_rad])}\n')
    
    # 僅查看設定範圍對應的 rad/s
    mask = omega < freq_limit_rad
    omega_plot = omega[mask]
    magnitude_plot = magnitude[mask]

    # 找峰值
    peaks, _ = scipy.signal.find_peaks(magnitude_plot, height=np.max(magnitude_plot)*0.1, distance=10)
    peak_freqs = omega_plot[peaks]
    peak_mags = magnitude_plot[peaks]

    # 只取最高峰值
    sorted_idx = np.argsort(peak_mags)[::-1][:1]
    top_peaks = peaks[sorted_idx]
    top_freqs = peak_freqs[sorted_idx]
    top_mags = peak_mags[sorted_idx]

    # 畫圖
    plt.figure(figsize=(14, 8))

    plt.subplot(2, 1, 1)
    time_axis = np.arange(len(all_errors_filtered)) * Ts + time_offset
    plt.plot(time_axis, all_errors_filtered)
    plt.xlabel('Time (s)')
    plt.ylabel('Error')
    title_parts = ['Full Error Time Domain']
    if TIME_START or TIME_END:
        title_parts.append(f'({time_offset:.1f}~{time_offset + len(all_errors_filtered)*Ts:.1f}s)')
    if HIGHPASS_FREQ_HZ:
        title_parts.append(f'HP {HIGHPASS_FREQ_HZ} Hz')
    plt.title(' '.join(title_parts))
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(omega_plot, magnitude_plot)
    plt.plot(top_freqs, top_mags, 'r*', markersize=12)
    plt.xlabel('Frequency (rad/s)')
    plt.ylabel('Magnitude')
    fft_title_parts = ['Full Error FFT Spectrum']
    if TIME_START or TIME_END:
        fft_title_parts.append(f'({time_offset:.1f}~{time_offset + len(all_errors_filtered)*Ts:.1f}s)')
    if HIGHPASS_FREQ_HZ:
        fft_title_parts.append(f'HP {HIGHPASS_FREQ_HZ} Hz')
    plt.title(' '.join(fft_title_parts))
    plt.xlim([0, freq_limit_rad])
    plt.ylim([0, np.max(magnitude_plot) * 1.2])
    for f, m in zip(top_freqs, top_mags):
        freq_hz = f / (2 * np.pi)
        label = f'{f:.1f} rad/s ({freq_hz:.1f} Hz)'
        plt.text(f, m*1.08, label, ha='center', fontsize=14, color='red', fontweight='bold')
    plt.grid(True)

    plt.tight_layout()

    output_dir = os.path.dirname(file_path)
    output_path = os.path.join(output_dir, 'error_full_fft.png')
    plt.savefig(output_path, dpi=150)

    print('最高峰值:')
    for i, (f, m) in enumerate(zip(top_freqs, top_mags)):
        print(f'{f:.2f} rad/s ({f/(2 * np.pi):.2f} Hz) - Magnitude: {m:.2f}')
    print(f'\nImage saved: {output_path}')

    plt.show()


if __name__ == '__main__':
    main()
