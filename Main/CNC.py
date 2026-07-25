import os
import sys
import control as ctrl
from control.matlab import lsim
from gurobipy import *
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.signal import tf2zpk


# ============================================================
# 繪圖尺寸標準（全專案共用，Plot_Exp_Data.py / Toolbox.py 由此引用）
# 像素需被 16 整除以相容影片編碼（dpi=100）
# ============================================================
FIG_SIZE_SINGLE = (7.68, 5.76)    # 768x576  單張標準圖
FIG_SIZE_WIDE = (12, 6.08)        # 1200x608 寬版圖（影片影格比例）
FIG_SIZE_MULTI = (10.24, 10.24)   # 1024x1024 多合一/方形圖

# 字體大小（圖片越小，字體相對越大）
FONT_TITLE = 24    # 圖表標題
FONT_LABEL = 20    # 座標軸標籤
FONT_TICK = 18     # 刻度數字
FONT_LEGEND = 18   # 圖例
FONT_TEXT = 18     # 標註文字（plt.text 等）


def apply_plot_style():
    """套用全專案統一的 matplotlib 字體樣式（Plot_Exp_Data.py / Toolbox.py 共用）。"""
    plt.rcParams.update({
        'axes.titlesize': FONT_TITLE,
        'axes.labelsize': FONT_LABEL,
        'xtick.labelsize': FONT_TICK,
        'ytick.labelsize': FONT_TICK,
        'legend.fontsize': FONT_LEGEND,
        'font.size': FONT_TEXT,
    })


def SimulateResponse(path, CC, plant, X0=0, Ts=0.001):
    """模擬閉迴路系統響應，回傳 (下一時間步狀態, 誤差um, 輸出響應)。"""
    # 將 plant 轉換為狀態空間形式
    plant_ss = ctrl.tf2ss(plant)
    
    # 計算開迴路轉移函數
    OLoop = ctrl.ss2tf(CC * plant_ss)
    num = OLoop.num[0][0]
    den = OLoop.den[0][0]
    
    # 確保分子分母長度一致（零填充）
    if len(num) < len(den):
        num = np.pad(num, (len(den) - len(num), 0), 'constant')
    
    # 建立閉迴路轉移函數: CLoop = OLoop / (1 + OLoop)
    CLoop = ctrl.TransferFunction(num, den + num, Ts)
    CLoop_ss = ctrl.tf2ss(CLoop)
    
    # 模擬系統響應
    T = np.arange(0, len(path) * Ts, Ts)
    _, y, X = ctrl.forced_response(CLoop_ss, T, path, X0, return_x=True)
    
    # 計算下一時間步的狀態: x[k+1] = A*x[k] + B*u[k]
    X_next = CLoop_ss.A @ X[:, -1].reshape(-1, 1) + CLoop_ss.B * path[-1]
    
    # 計算追蹤誤差（轉換為微米）
    error_um = (path - y) * 1000
    
    return X_next, error_um, y

def find_Wgc(CC, plant, Ts):
    """用切比雪夫多項式法計算增益交越頻率 Wgc (rad/s)。"""
    # 計算開迴路轉移函數並最小化
    OLoop = ctrl.minreal(ctrl.ss2tf(CC * plant), tol=1e-3, verbose=False)

    # 提取分子分母係數
    a = OLoop.den[0][0]  # 分母係數
    b = OLoop.num[0][0]  # 分子係數
    n = len(a) - 1       # 系統階數
    
    # 確保分子長度與分母一致
    if len(b) < n + 1:
        b = np.pad(b, (n + 1 - len(b), 0), 'constant')
    
    # 建立第一類切比雪夫多項式係數矩陣
    # T_k(cos(w)) = cos(k*w)，用於將頻率響應轉換為多項式形式
    Cheby = np.zeros((n + 1, n + 1))
    if n == 1:
        Cheby[:, 0], Cheby[:, 1] = [0, 1], [1, 0]
    else:
        Cheby[n, 0] = 1
        Cheby[n - 1, 1] = 1
        for i in range(2, n + 1):
            # 遞迴關係: T_n(x) = 2x*T_{n-1}(x) - T_{n-2}(x)
            Cheby[:, i] = np.append(2 * Cheby[1:n + 1, i - 1], 0) - Cheby[:, i - 2]
    
    # 計算 |A(e^{jw})|^2 和 |B(e^{jw})|^2 的多項式係數
    # 利用 rho = [1, e^{-jw}, ..., e^{-jnw}] 展開
    P_den = np.zeros(n + 1)  # |A(e^{jw})|^2
    P_num = np.zeros(n + 1)  # |B(e^{jw})|^2
    for i in range(n + 1):
        for k in range(n + 1):
            P_den += a[i] * a[k] * Cheby[:, abs(k - i)]
            P_num += b[i] * b[k] * Cheby[:, abs(k - i)]
    
    # 求解 |B|^2 - |A|^2 = 0 的根（即 |OLoop| = 1）
    R = P_den - P_num
    roots = np.roots(R)
    
    # 篩選實根且在 [-1, 1] 範圍內（對應 cos(w) 的有效範圍）
    real_roots = roots[np.isreal(roots)]
    valid_roots = np.real(real_roots[np.abs(real_roots) < 1])
    
    # 取最大根對應的頻率（最小的 w）
    max_cos_w = np.max(valid_roots)
    w_normalized = np.arccos(max_cos_w)  # 正規化角頻率 (0 ~ pi)
    
    # 轉換為實際角頻率 (rad/s)
    Wgc = w_normalized / Ts
    
    return Wgc

def find_resonance(CC, plant, Ts, path_section, error):
    """
    從誤差頻譜辨識高頻共振點 (頻率 > Wgc 且振幅 > 5倍輸出平均)。
    回傳 (共振頻率 rad/s, 共振振幅, 共振增益 dB)。
    """
    def compute_fft(data):
        """計算單邊傅立葉頻譜（含 Hanning 窗）"""
        data_len = len(data)
        window = np.hanning(data_len)
        windowed_data = window * data
        
        N = int(1 / Ts)  # FFT 點數 = 取樣頻率，頻率解析度為 1 Hz
        spectrum = scipy.fft.fft(windowed_data, N)
        freq = scipy.fft.fftfreq(N, Ts)
        
        positive_idx = N // 2
        return freq[:positive_idx], np.abs(spectrum[:positive_idx]) / data_len
    
    # 取得增益交越頻率
    Wgc = find_Wgc(CC, plant, Ts)
    Fgc = Wgc / (2 * np.pi)  # 轉換為 Hz
    
    # 計算輸出訊號
    output = path_section - error
    
    # 去除誤差的直流偏移
    error_ac = error - error[0]
    
    # 計算頻譜
    freq_axis, error_spectrum = compute_fft(error_ac)
    _, output_spectrum = compute_fft(output)
    
    # 找出誤差頻譜的峰值位置
    peak_indices, _ = scipy.signal.find_peaks(error_spectrum)
    
    # 篩選高頻共振：頻率 > Fgc 且 振幅 > 5倍輸出平均振幅
    output_mean = np.mean(output_spectrum)
    is_high_freq = freq_axis[peak_indices] > Fgc
    is_significant = error_spectrum[peak_indices] > output_mean * 5
    valid_mask = is_high_freq & is_significant
    
    valid_peaks = peak_indices[valid_mask]
    
    # 提取共振特徵
    resonance_freq = freq_axis[valid_peaks] * 2 * np.pi      # 轉為 rad/s
    resonance_mag = error_spectrum[valid_peaks] * 2 * np.pi  # 轉為 rad/s 域振幅
    
    # 計算共振增益 (dB)
    error_mag_at_peaks = error_spectrum[valid_peaks]
    output_mag_at_peaks = output_spectrum[valid_peaks]
    resonance_gain_dB = 20 * np.log10(error_mag_at_peaks / output_mag_at_peaks)
    
    return resonance_freq, resonance_mag, resonance_gain_dB

class CNCModel:
    """CNC 馬達模型類別，提供系統鑑別模型、測試模型、不確定性模型等。"""

    def __init__(self, axis, Ts):
        self.Ts = Ts
        self.axis = axis
        self.integrator = ctrl.TransferFunction([Ts, 0], [1, -1], Ts)  # 離散積分器

    def ID_Plant(self):
        """回傳系統鑑別後的 Plant 模型 (v2v: 速度轉速度, v2p: 速度轉位置)。"""
        # X軸馬達轉移函數
        num_x = [0, 0.0410789388950551, 0.116567016168533]
        den_x = [1, -1.41353788543924, 0.566877911191563]
        Pv_x = ctrl.TransferFunction(num_x, den_x, self.Ts)
        rpm2mms_x = 10 / 60  # rpm 轉 mm/s
        Px = Pv_x * rpm2mms_x * self.integrator
        
        # Z軸馬達轉移函數 (目前未使用)
        num_z = [0, 0.0973217504156460, -0.209580502231482, 0.151454774135651, -0.0370139184925609]
        den_z = [1, -3.44700230387150, 4.55138855100522, -2.73113485001183, 0.628907804339994]
        Pv_z = ctrl.TransferFunction(num_z, den_z, self.Ts)
        rpm2mms_z = 12 / 60
        Pz = Pv_z * rpm2mms_z * self.integrator
        
        v2v = {'x': Pv_x, 'z': Pv_z}
        v2p = {'x': Px, 'z': Pz}
        return {'v2v': v2v[self.axis], 'v2p': v2p[self.axis], 'Ts': self.Ts}

    def test_Plant(self, omega=600, zeta=0.01):
        """回傳帶有固定共振點的測試模型。"""
        base_plant = self.ID_Plant()
        v2p = base_plant['v2p']

        # 共振峰 1：600 rad/s
        res1 = ctrl.TransferFunction(
            [1, 8 * zeta * omega, omega**2],
            [1, 2 * zeta * omega, omega**2]
        )
        # 共振峰 2：800 rad/s
        omega2 = 800
        res2 = ctrl.TransferFunction(
            [1, 7 * zeta * omega2, omega2**2],
            [1, 2 * zeta * omega2, omega2**2]
        )
        # 串接兩個共振後離散化
        resonance_tf = ctrl.sample_system(res1 , self.Ts)#* res2
        v2p = v2p * resonance_tf
        v2v = base_plant['v2v'] * resonance_tf
        return {'v2p': v2p, 'v2v': v2v, 'Ts': self.Ts}

    def BUE_Plant(self):
        """從 Delta_Data.mat 隨機選取一組不確定性模型 (Base Uncertainty Ensemble)。"""
        def cancel_pole_zero(z, p, tol=1e-3):
            """自動 pole-zero 對消"""
            z_new, p_new = list(z), list(p)
            for zero in z:
                diffs = [abs(zero - pole) for pole in p_new]
                if diffs and min(diffs) < tol:
                    idx = diffs.index(min(diffs))
                    z_new.remove(zero)
                    del p_new[idx]
            return np.array(z_new, dtype=complex), np.array(p_new, dtype=complex)
        
        # 取得基礎模型的 zpk (v2p 與 v2v 同步)
        base_plant = self.ID_Plant()
        num_p, den_p = base_plant["v2p"].num[0][0], base_plant["v2p"].den[0][0]
        IDz_p, IDp_p, IDk_p = tf2zpk(num_p, den_p)
        num_v, den_v = base_plant["v2v"].num[0][0], base_plant["v2v"].den[0][0]
        IDz_v, IDp_v, IDk_v = tf2zpk(num_v, den_v)

        # 載入不確定性資料
        data = scipy.io.loadmat('Delta_Data.mat')
        z_all = data['z_all'].squeeze()
        p_all = data['p_all'].squeeze()
        k_all = data['k_all'].squeeze()
        Ts = float(data['Ts'])

        # 合成所有不確定性模型 (v2p 與 v2v 套用相同 delta)
        v2ps = []
        v2vs = []
        for i in range(len(z_all)):
            deltaz = z_all[i].squeeze().astype(complex)
            deltap = p_all[i].squeeze().astype(complex)
            deltak = k_all[i].item()
            z_p = np.concatenate([deltaz, IDz_p])
            p_p = np.concatenate([deltap, IDp_p])
            z_v = np.concatenate([deltaz, IDz_v])
            p_v = np.concatenate([deltap, IDp_v])
            z_new_p, p_new_p = cancel_pole_zero(z_p, p_p, tol=1e-2)
            z_new_v, p_new_v = cancel_pole_zero(z_v, p_v, tol=1e-2)
            v2ps.append(ctrl.zpk(z_new_p, p_new_p, deltak * IDk_p, Ts))
            v2vs.append(ctrl.zpk(z_new_v, p_new_v, deltak * IDk_v, Ts))

        # 隨機選擇同一組
        idx = np.random.randint(0, len(z_all))
        return {'v2p': v2ps[idx], 'v2v': v2vs[idx], 'Ts': self.Ts}

    def PRE_Plant(self, min_omega=300, max_omega=1000):
        """回傳帶有隨機共振點的不確定性模型 (Perturbed Resonant Ensemble，頻率範圍: min_omega ~ max_omega rad/s)。"""
        base_plant = self.BUE_Plant()
        v2p = base_plant['v2p']
        v2v = base_plant['v2v']
        # 隨機頻率 (低頻機率較高)
        omega_candidates = np.linspace(min_omega, max_omega, num=500)
        alpha = np.log(5) / (max_omega - min_omega)
        weights = np.exp(-alpha * (omega_candidates - min_omega))
        weights /= np.sum(weights)
        omega = np.random.choice(omega_candidates, p=weights)
        
        # 隨機阻尼比
        zeta = np.random.uniform(0.005, 0.05)
        
        # 隨機峰值大小 (頻率越高，峰值越大)
        norm_freq = (omega - min_omega) / (max_omega - min_omega)
        gain_min = 5 * norm_freq * np.exp(norm_freq * 2) + 2
        gain_max = 10 * norm_freq * np.exp(norm_freq * 2) + 2
        gain = np.random.uniform(gain_min, gain_max)
        
        # 添加共振點
        resonance_tf = ctrl.TransferFunction(
            [1, gain * zeta * omega, omega**2],
            [1, 2 * zeta * omega, omega**2]
        )
        resonance_tf = ctrl.sample_system(resonance_tf, self.Ts)
        v2p = v2p * resonance_tf
        v2v = v2v * resonance_tf
        return {'v2p': v2p, 'v2v': v2v, 'Ts': self.Ts}


class PathModel:
    """路徑生成類別，提供各種測試與訓練用路徑。"""
    
    def __init__(self, Ts, path_time=15):
        self.Ts = Ts
        self.path_time = path_time

    def chirp_path(self, f_start=0, f_end=1, amplitude=1):
        """生成線性 chirp 訊號 (頻率從 f_start 到 f_end Hz)。"""
        t = np.arange(0, self.path_time, self.Ts)
        return scipy.signal.chirp(t, f0=f_start, f1=f_end, t1=self.path_time, 
                                   method='linear', phi=-90) * amplitude

    def test_path(self):
        """回傳 0~1 Hz 的 chirp 測試路徑。"""
        return self.chirp_path(f_end=1)

    def test_path2(self):
        """回傳 0~8 Hz 的 chirp 測試路徑。"""
        return self.chirp_path(f_end=8)

    def training_path(self):
        """回傳 20 條訓練路徑 (混合正弦波與 chirp)。"""
        t = np.arange(0, self.path_time, self.Ts)
        pt = self.path_time
        
        # 混合頻率正弦波 (10條)
        sine_paths = [
            0.55*np.cos(2*np.pi*0.1*t + np.pi/9) + 0.35*np.sin(2*np.pi*0.3*t + np.pi/5) + 0.1,
            0.6*np.cos(2*np.pi*0.4*t + np.pi/7) + 0.25*np.cos(2*np.pi*0.8*t + np.pi/3) - 0.1,
            0.4*np.sin(2*np.pi*1.0*t + np.pi/4) + 0.35*np.cos(2*np.pi*1.8*t + np.pi/6) + 0.25*np.sin(2*np.pi*0.6*t + np.pi/2),
            0.5*np.cos(2*np.pi*3.0*t + np.pi/2) + 0.4*np.sin(2*np.pi*5.5*t + np.pi/9) - 0.15,
            0.45*np.sin(2*np.pi*0.2*t + np.pi/3) + 0.4*np.cos(2*np.pi*6.0*t + np.pi/10) + 0.15,
            0.7*np.cos(2*np.pi*0.05*t + np.pi/8) + 0.2,
            0.35*np.sin(2*np.pi*1.0*t + np.pi/5) + 0.35*np.sin(2*np.pi*1.3*t + np.pi/7) + 0.25*np.cos(2*np.pi*1.6*t + np.pi/2),
            0.5*np.cos(2*np.pi*2.5*t + np.pi/6) + 0.3*np.cos(2*np.pi*5.0*t + np.pi/4) - 0.1,
            0.4*np.sin(2*np.pi*7.0*t + np.pi/3) + 0.3*np.cos(2*np.pi*9.0*t + np.pi/10),
            0.6*np.sin(2*np.pi*4.5*t + np.pi/2) + 0.2,
        ]
        
        # Chirp 訊號 (10條，涵蓋不同頻段)
        chirp_paths = [
            0.8*scipy.signal.chirp(t, f0=0.02, f1=0.2, t1=pt, method='linear', phi=-40) + 0.05,   # 超低頻
            0.8*scipy.signal.chirp(t, f0=0.1,  f1=0.8, t1=pt, method='linear', phi=-70) - 0.05,   # 低頻
            0.8*scipy.signal.chirp(t, f0=0.3,  f1=1.5, t1=pt, method='linear', phi=-20),          # 低～中
            0.8*scipy.signal.chirp(t, f0=0.8,  f1=3.0, t1=pt, method='linear', phi=-90) + 0.05,   # 中
            0.8*scipy.signal.chirp(t, f0=2.0,  f1=4.0, t1=pt, method='linear', phi=-10),          # 中
            0.8*scipy.signal.chirp(t, f0=3.0,  f1=7.0, t1=pt, method='linear', phi=-60) - 0.05,   # 中～高
            0.8*scipy.signal.chirp(t, f0=0.0,  f1=9.0, t1=pt, method='linear', phi=-110) + 0.05,  # 高
            0.8*scipy.signal.chirp(t, f0=0.5,  f1=0.6, t1=pt, method='linear', phi=-30),          # 低頻窄帶
            0.8*scipy.signal.chirp(t, f0=1.5,  f1=0.2, t1=pt, method='linear', phi=-50) + 0.05,   # 反向低～中
            0.8*scipy.signal.chirp(t, f0=9.0,  f1=3.0, t1=pt, method='linear', phi=-75) - 0.05,   # 反向高→中
        ]
        
        return sine_paths + chirp_paths

    def up_down_chirp(self, fmin=0, fmax=8, amplitude=1.0):
        """生成上下掃頻 chirp (時間為 2 倍 path_time)。"""
        path_time = self.path_time * 2
        t = np.arange(0, path_time, self.Ts)
        
        # 頻率先升後降的相位函數
        phi = 2*np.pi * (fmin*t + (fmax - fmin) * (0.5*t - (path_time/(4*np.pi)) * np.sin(2*np.pi*t/path_time)))
        return amplitude * np.sin(phi)


class Costfunction:
    """
    QCQP 控制器設計類別。
    透過互質分解與線性分式轉換，將控制器設計問題轉換為 QCQP，
    以時域誤差最小化為目標，頻域増益限制為約束條件。
    """
    
    def __init__(self, CNC_parameter, polegain, plant, path, pdl, numFC, numLowFreq, manual_FC=-1):
        """初始化 QCQP 求解器，建立互質分解與 LFT 架構。"""
        
        def coprime_factorization_ss(plant, polegain):
            """互質分解：將 Plant 分解為 M, N, X, W 四個子系統"""
            Ts = plant['Ts']
            poles = ctrl.pole(plant['v2p'])
            G = ctrl.tf2ss(plant['v2p'])
            
            # 重新指定極點位置（將接近單位圓的極點向內縮）
            adjusted_poles = []
            for pole in poles:
                if abs(pole) > 0.99:
                    adjusted_poles.append(pole * polegain)
                else:
                    adjusted_poles.append(pole)
            
            # 極點配置法計算回授增益 F 與觀測器增益 H
            F = ctrl.place(G.A, -1*G.B, adjusted_poles)
            H = ctrl.place(G.A.T, -1*G.C.T, adjusted_poles)
            
            # 根據 F 和 H 計算互質分解的 M, N, X, W
            M = ctrl.ss(G.A + G.B @ F, G.B, F, 1, Ts)
            N = ctrl.ss(G.A + G.B @ F, G.B, G.C + G.D @ F, G.D, Ts)
            X = ctrl.ss(G.A + H.T @ G.C, H.T, F, np.zeros_like(G.D), Ts)
            W = ctrl.ss(G.A + H.T @ G.C, -G.B - H.T @ G.D, F, np.ones_like(G.D), Ts)

            # 更新 plant 字典並返回
            plant['G'] = G
            plant['F'] = F
            plant['H'] = H
            plant['assignedpole'] = adjusted_poles
            plant['M'] = M
            plant['N'] = N
            plant['X'] = X
            plant['W'] = W
            return plant

        def linear_fractional_transformation(plant):
            """線性分式轉換：建立 Jy 系統供 Q 參數化使用"""
            G = plant['G']
            F = plant['F']
            H = plant['H']
            L = H.T
            
            # 組合 Jy 的狀態空間矩陣
            JA = G.A + G.B @ F + L @ G.C + L @ G.D @ F
            JB = np.concatenate((L, G.B + L @ G.D), axis=1)
            JC = np.concatenate((F, (G.C + G.D @ F)), axis=0)
            n = G.D.shape[0]
            I = np.eye(n)
            JD = np.vstack([np.hstack([np.zeros((n, n)), I]), np.hstack([I, G.D])])
        
            Jy = ctrl.ss(JA, JB, JC, JD, plant['Ts'])
            plant['Jy'] = Jy
            return plant

        def use_central_controller():
            self.CC=self.LFTExpandedSS(np.zeros((self.parameter.Lq, 1)))
            OLoop=ctrl.minreal( ctrl.ss2tf(self.CC * self.plant['G']), tol=1e-3, verbose=False)
            maxOmega=np.pi*(1/self.plant['Ts'])
            minOmega=0.1
            mag, _, omega = ctrl.bode(OLoop, dB=True, omega_limits=[minOmega, maxOmega], plot=False)
            gain_crossover_idx = np.where(np.isclose(20*np.log10(mag), 0, atol=0.5))[0]
            if len(gain_crossover_idx) > 0:
                Wgc = omega[gain_crossover_idx[0]]  # 選擇第一個交越點
            else:
                sys.exit("找不到Central Controller 的Wgc")
            max_omega_dB = 20 * np.log10(maxOmega)
            min_omega_dB = 20 * np.log10(minOmega)
            Wgc_dB = 20 * np.log10(Wgc)
            # 生成限制點
            lowOmegas = 10**(np.linspace(min_omega_dB, Wgc_dB, numLowFreq, endpoint=False) /20)
            highOmegas = 10**(np.linspace(Wgc_dB, max_omega_dB, numFC-numLowFreq + 2, endpoint=False)[1:-1]  /20)
            scalMag=0.9
            while True:
                lowMag, _,_ = ctrl.bode(OLoop, dB=True, omega=lowOmegas, plot=False)
                highMag, _,_ = ctrl.bode(OLoop, dB=True, omega=highOmegas, plot=False)
                lowMag=10**((20*np.log10(lowMag)*scalMag)/20)
                highMag=10**((20*np.log10(highMag)*scalMag)/20)
                FC = np.vstack([
                    np.hstack([lowOmegas, highOmegas]),  # 第一行：限制點頻率
                    np.hstack([lowMag, highMag])  # 第二行：對應的增益
                    ]).T  # 轉置成 (10,2)
                '''
                OLoop=ctrl.minreal( ctrl.ss2tf(self.CC * self.plant['v2p']), tol=1e-3, verbose=False)
                mag,_,oma=ctrl.bode(OLoop, dB=True, omega_limits=[1e-2, 3e3], plot=False)#OLoop波德圖
                plt.plot(oma,20*np.log10(mag),color='b')
                sorted_indices = np.argsort(FC[:, 0])  #排序FC的索引
                FC = FC[sorted_indices]  # 按照索引排序FC
                plt.scatter(FC[:,0], 20*np.log10(FC[:,1]), color='r')
                plt.plot(FC[:,0], 20*np.log10(FC[:,1]), color='r')
                plt.grid()
                plt.xscale('log')
                plt.xlim(1e-2, 1e4)
                plt.ylim(-70,70)
                plt.xlabel("Frequency (rad/s)")
                plt.ylabel("Magnitude(dB)")
                plt.show()
                #'''
                status, _ = self.optimizationcvx(path, 0, FC)
                if status == "Solved":
                    self.last_solved_FC = FC.copy()
                    break
                else:
                    scalMag = scalMag * 0.9

        # 初始化成員變數
        self.parameter = CNC_parameter
        self.plant = linear_fractional_transformation(coprime_factorization_ss(plant, polegain))
        self.path = path
        self.pdl = pdl
        self.numLowFreq = numLowFreq
        self.resonanceTable = np.zeros((3, 1))

        # 建立 Gurobi 模型
        self.model = Model("QCQP")
        self.model.setParam('OutputFlag', 0)
         
        # 初始化控制器
        if manual_FC != -1:
            self.set_controller(manual_FC)
        else:
            use_central_controller()
         
        self.initialize()

    def initialize(self):
        """每輪開始時重置狀態。"""
        self.X0_hat = 0
        self.resonanceTable = np.zeros((5, 1))

    def set_controller(self, FC):
        """手動設定頻域限制點 FC，若無解則使用中央控制器。"""
        state, Q = self.optimizationcvx(self.path, 0, FC.copy())
        if state == "Solved":
            self.last_solved_FC = FC
            self.CC = self.LFTExpandedSS(Q)
        else:
            self.last_solved_FC = FC
            self.CC = self.LFTExpandedSS(np.zeros((self.parameter.Lq, 1)))
            print("Manual FC 無解，使用 Central Controller")
        
    def optimizationcvx(self, path, path_index, FC):
        """
        求解 QCQP 優化問題。
        目標函數: 時域誤差加權平方和最小化
        限制條件: 頻域增益上下界
        回傳: (status, Q參數)
        """
        def time_domain_coefficient(path, path_index):
            """計算 QCQP 的時域係數 (Tcoe2, Tcoe1, Tcoe0)"""
            Ts = self.plant['Ts']
            M = self.plant['M']
            N = self.plant['N']
            W = self.plant['W']
            L = self.parameter.Lq

            # 取得路徑經過各個狀態矩陣後的輸出
            path_seg = path[path_index:path_index + self.pdl]
            T_sim = np.arange(len(path_seg)) * Ts
            ym, _, _ = lsim(M, U=path_seg, T=T_sim)
            zk, _, _ = lsim(W, U=ym, T=T_sim)
            vk, _, _ = lsim(N, U=ym, T=T_sim)
         
            # 初始化各項係數
            Tphi = np.zeros((L, 1))
            Tcoe2_t = np.zeros((L, L))
            Tcoe1_t = np.zeros((1, L))
            Tcoe0_t = 0
        
            # 計算係數 (加權系數 0.7^k)
            for k in range(len(vk)):
                wk = 0.7 ** k
                Tphi = np.vstack((vk[k], Tphi[:-1]))
                Tcoe2_t += wk * np.outer(Tphi, Tphi)
                Tcoe1_t += wk * zk[k] * Tphi.T
                Tcoe0_t += wk * zk[k] ** 2

            return Tcoe2_t, Tcoe1_t, Tcoe0_t

        def freq_domain_coefficient(FC):
            """計算 QCQP 的頻域限制係數 (Omega, beta, gamma)"""
            L = self.parameter.Lq
            Nc = FC.shape[0]  # 限制條件個數
            alpha = np.ones(Nc)
            beta = np.zeros((L, Nc))
            Omega = np.zeros((L, L, Nc))
            gamma = np.zeros(Nc)
            weight = np.zeros(Nc)
            w0 = 100000
            
            N = self.plant['N']
            M = self.plant['M']
            X = self.plant['X']
            W = self.plant['W']
            P = self.plant['v2p']
            Ts = self.plant['Ts']

            # 避開 ki=1 的奇異點
            tol = 1e-11
            eps = 1e-10
            mask_close = np.isclose(FC[:, 1], 1.0, atol=tol)
            FC[mask_close & (FC[:, 1] > 1.0), 1] -= eps
            FC[mask_close & ~(FC[:, 1] > 1.0), 1] += eps

            wi = FC[:, 0]  # 頻率點 (rad/s)
            ki = FC[:, 1]  # 增益限制 (>1: 下界, <1: 上界)
            
            # 計算各系統在限制頻率點的頻率響應
            magP, _, _ = ctrl.bode(P, wi, plot=False)
            magM, phaseM, _ = ctrl.bode(M, wi, plot=False)
            magN, phaseN, _ = ctrl.bode(N, wi, plot=False)
            magX, phaseX, _ = ctrl.bode(X, wi, plot=False)
            magW, phaseW, _ = ctrl.bode(W, wi, plot=False)
            
            # 將幅度和相位轉換為複數形式
            phaseM, phaseN = np.rad2deg(phaseM), np.rad2deg(phaseN)
            phaseX, phaseW = np.rad2deg(phaseX), np.rad2deg(phaseW)
            valueM = magM * (np.cos(np.deg2rad(phaseM)) + 1j * np.sin(np.deg2rad(phaseM)))
            valueN = magN * (np.cos(np.deg2rad(phaseN)) + 1j * np.sin(np.deg2rad(phaseN)))
            valueX = magX * (np.cos(np.deg2rad(phaseX)) + 1j * np.sin(np.deg2rad(phaseX)))
            valueW = magW * (np.cos(np.deg2rad(phaseW)) + 1j * np.sin(np.deg2rad(phaseW)))
        
            # 建立每個頻率點的限制矩陣
            for i in range(Nc):
                weight[i] = 0.0000005
                rho = np.zeros((L, 1), dtype=complex)
                omevector = np.zeros((L, 1))
                for j in range(L):
                    rho[j] = np.exp(-j * wi[i] * 1j * Ts)
                    omevector[j] = np.cos(j * wi[i] * Ts)
                
                Omega[:, :, i] = scipy.linalg.toeplitz(omevector)
                b = np.real(np.conj(valueW[i]) * valueN[i] * rho).ravel()
                c = np.real(np.conj(valueX[i]) * valueM[i] * rho).ravel()

                beta[:, i] = (-2 * (ki[i]**2 * b + magP[i]**2 * c)) / ((ki[i]**2 - 1) * magN[i]**2)
                gamma[i] = ((ki[i]**2 * np.abs(valueW[i])**2) - (magP[i]**2 * np.abs(valueX[i])**2)) / ((ki[i]**2 - 1) * magN[i]**2)

            return {
                'alpha': alpha, 'beta': beta, 'Omega': Omega,
                'Nc': Nc, 'gamma': gamma, 'weight': weight, 'w0': w0
            }

        # 計算時域與頻域係數
        Lq = self.parameter.Lq
        model = self.model
        model.remove(model.getQConstrs())
        Tcoe2_t, Tcoe1_t, Tcoe0_t = time_domain_coefficient(path, path_index)
        allFC = freq_domain_coefficient(FC)
        q = model.addMVar(shape=Lq, vtype=GRB.CONTINUOUS, name="q")
        
        # 設定目標函數: min q'*Tcoe2*q - 2*Tcoe1*q + Tcoe0
        obj = q @ Tcoe2_t @ q - 2 * Tcoe1_t @ q + Tcoe0_t
        model.setObjective(obj, GRB.MINIMIZE)

        # 加入頻域限制條件: q'*Omega*q + beta'*q + gamma <= 0
        for fi in range(allFC['Nc']):
            Omega_i = allFC['Omega'][:, :, fi]
            beta_i = allFC['beta'][:, fi]
            gamma_i = allFC['gamma'][fi]
            model.addConstr(q @ Omega_i @ q + beta_i @ q + gamma_i <= 0)
        
        # 求解
        model.optimize()
        if model.status == GRB.OPTIMAL:
            status = "Solved"
            solution = np.array(q.x).reshape((Lq, 1))
        else:
            status = "Infeasible"
            solution = 0
        return status, solution

    def LFTExpandedSS(self, Qnum):
        """將 Q 參數與 Jy 組合成控制器狀態空間。"""
        lq = self.parameter.Lq
        J = self.plant['Jy']
        
        # 建立 Q 的狀態空間矩陣
        qa1 = np.zeros(lq - 1)
        qa2 = np.zeros(lq - 1)
        qa1[1] = 1
        QA = scipy.linalg.toeplitz(qa1, qa2)
        QB = np.zeros((lq - 1, 1))
        QB[0] = 1
        QC = np.array(Qnum[1:]).reshape(1, -1)
        QD = Qnum[0]
        
        # LFT 展開計算
        F = 1 / (1 - QD * J.D[1, 1])
        Dk = F * QD
        Ck = J.C[0, :] + F * QD * J.C[1, :]
        Ck = np.hstack([Ck, F * QC])
        Bk = J.B[:, 0] + J.B[:, 1] * F * QD
        Bk = np.vstack([Bk, QB + QB * J.D[1, 1] * F * QD])
        
        Ak11 = J.A[:, :] + J.B[:, 1] * F * QD * J.C[1, :]
        Ak12 = J.B[:, 1] * F * QC
        Ak21 = QB * J.C[1, :] + QB * J.D[1, 1] * F * QD * J.C[1, :]
        Ak22 = QA + QB * J.D[1, 1] * F * QC
        Ak = np.vstack([
            np.hstack([Ak11, Ak12]),
            np.hstack([Ak21, Ak22])
        ])
        
        CC = ctrl.ss(Ak, Bk, Ck, Dk, self.plant['Ts'])
        return CC

    def switch_controller(self, path, path_index, FC, ek):
        """
        PPO 主調用入口：根據 FC 求解 QCQP 並執行共振壓制。
        回傳: (status, CC, ek_hat, manual_add_FC)
        """
        # 遞減共振紀錄表中所有延遲步數 > 0 的項目
        for col in range(self.resonanceTable.shape[1]):
            if self.resonanceTable[0, col] == 0:
                break
            if self.resonanceTable[3, col] is not None and self.resonanceTable[3, col] > 0:
                self.resonanceTable[3, col] -= 1
        
        def suppress_resonance(original_status, Q, FC, path, path_index, ek, 
                               freq_tolerance=30, reduce_mag=10, increase_mag=5):
            """偵測並壓制高頻共振"""
            
            def find_working_target(freq, start_target):
                """搜尋可行的壓制目標增益"""
                for target in range(start_target - reduce_mag, start_target, increase_mag):
                    test_FC = np.vstack([FC, [freq, 10 ** (target / 20)]])
                    test_FC = test_FC[np.argsort(test_FC[:, 0])]
                    status, _ = self.optimizationcvx(path, path_index, test_FC)
                    if status == "Solved":
                        return target, True
                return None, False

            manual_add_FC = np.empty((0, 2))

            # 沒有誤差資料或 QCQP 無解時直接返回
            if np.sum(ek) == 0 or original_status != "Solved":
                return original_status, Q, manual_add_FC

            # 找出潛在共振頻率
            path_segment = path[path_index:path_index + self.pdl]
            CC = self.LFTExpandedSS(Q)
            OLoop = ctrl.minreal(ctrl.ss2tf(CC * self.plant["v2p"]), tol=1e-3, verbose=False)
            resonance_freqs, _, _ = find_resonance(CC, self.plant["v2p"], self.plant['Ts'], path_segment, ek)
            resonance_freqs = np.array(resonance_freqs)
            
            # 比對共振紀錄表
            saved_freqs = self.resonanceTable[0, :]
            diff_matrix = np.abs(resonance_freqs[:, np.newaxis] - saved_freqs)
            match_indices = diff_matrix < freq_tolerance
            has_match = np.any(match_indices, axis=1)

            # 新增未登記的共振
            unsaved_freqs = resonance_freqs[~has_match]
            for freq in unsaved_freqs:
                insert_pos = np.searchsorted(self.resonanceTable[0, :], freq)
                new_col = np.array([
                    [freq],    # 頻率
                    [3],       # 檢查真共振步數（=3 對齊實驗；偵測後 +3 步才確認）
                    [None],    # Target
                    [1],       # 延遲步數
                    [True],    # 標記為可加入
                ])
                if self.resonanceTable[0, 0] == 0:
                    self.resonanceTable = new_col
                else:
                    self.resonanceTable = np.hstack((
                        self.resonanceTable[:, :insert_pos], 
                        new_col, 
                        self.resonanceTable[:, insert_pos:]
                    ))
            
            # 建立暫存共振表
            tempTable = self.resonanceTable.copy()

            # 重新計算匹配索引
            saved_freqs = tempTable[0, :]
            diff_matrix = np.abs(resonance_freqs[:, np.newaxis] - saved_freqs)
            match_indices = diff_matrix < freq_tolerance
            
            for i in range(len(resonance_freqs)):
                idx = np.where(match_indices[i, :])[0][0]
                if tempTable[1, idx] == 0:  # 確認為真共振
                    if tempTable[2, idx] is None:  # 第一次壓制
                        complex_mag, _, _ = ctrl.freqresp(OLoop, resonance_freqs[i])
                        loopgain = 0 if 20 * np.log10(np.abs(complex_mag)) > 0 else int(20 * np.log10(np.abs(complex_mag)))
                        target, success = find_working_target(resonance_freqs[i], loopgain)
                        if success:
                            tempTable[2, idx] = target
                        else:
                            tempTable[2, idx] = loopgain + 1
                            tempTable[3, idx] = 0
                    elif tempTable[3, idx] == 0:  # 延遲步數為0，仍有共振
                        old_target = tempTable[2, idx]
                        target, success = find_working_target(resonance_freqs[i], old_target)
                        if success:
                            tempTable[2, idx] = target
                            tempTable[3, idx] = 1
                else:
                    tempTable[1, idx] -= 1

            # 根據暫存表設定手動 FC
            for i in range(tempTable.shape[1]):
                if tempTable[0, i] == 0:
                    break
                target = tempTable[2, i]
                if target is not None:
                    freq = tempTable[0, i]
                    mag = 10 ** (target / 20)
                    manual_add_FC = np.vstack([manual_add_FC, [freq, mag]])

            # 合併 FC 並排序
            combined = np.vstack([FC, manual_add_FC])
            new_FC = combined[np.argsort(combined[:, 0])]

            # 最終 QCQP 求解
            status, newQ = self.optimizationcvx(path, path_index, new_FC)

            if status == "Solved":
                self.resonanceTable = tempTable
                return status, newQ, manual_add_FC
            else:
                return "semiSolved", Q, manual_add_FC

        status, Q = self.optimizationcvx(path, path_index, FC)
        path_segment = path[path_index:path_index + self.pdl]
        status, Q, manual_add_FC = suppress_resonance(status, Q, FC, path, path_index, ek)

        if status == "Solved":
            self.CC = self.LFTExpandedSS(Q)
            self.last_solved_FC = FC.copy()
            self.X0_hat, ek_hat, _ = SimulateResponse(path_segment, self.CC, self.plant['v2p'], self.X0_hat, self.plant['Ts'])
        else:
            self.X0_hat, ek_hat, _ = SimulateResponse(path_segment, self.CC, self.plant['v2p'], self.X0_hat, self.plant['Ts'])

        return status, self.CC, ek_hat, manual_add_FC

    def reward(self, FC, status, CC, ek, visual=0):
        """
        計算 PPO 獎勵函數。
        Solved: 基於誤差與 FC 分布均勻度
        Infeasible: 基於與上次可行 FC 的差距
        """
        def fc_uniformity(FC, Wgc):
            """計算 FC 在高頻區的對數間距均勻度"""
            Nc1 = FC.shape[0]
            Nc2 = np.sum(FC[:, 0] < Wgc)
            
            log_gaps = []
            for i in range(Nc1 - 1, Nc2 - 1, -1):
                if FC[i - 1, 0] <= 0 or FC[i, 0] <= 0:
                    continue
                log_gap = np.log10(FC[i, 0]) - np.log10(FC[i - 1, 0])
                log_gaps.append(log_gap)
            
            if len(log_gaps) < 2:
                return 0
            return max(log_gaps) - min(log_gaps)

        Wgc = find_Wgc(CC, self.plant["G"], self.plant["Ts"])
        parameter = self.parameter
        cost = 0
        
        if status == "Solved" or status == "semiSolved":
            # 誤差成本（加權）
            weights = 0.7 ** np.arange(len(ek))
            sum_error = np.sum(abs(ek) * weights)
            cost_sum_error = parameter.w_sumError * sum_error
            cost += cost_sum_error

            # FC 分布均勻度成本
            fc_dist = fc_uniformity(FC, Wgc)
            cost_fc_dist = parameter.w_FCfreq * fc_dist
            cost += cost_fc_dist

            # semiSolved 額外懲罰
            cost_Wgc = 0
            if status == "semiSolved":
                cost_Wgc = parameter.w_Wgc * Wgc
                cost += cost_Wgc

            reward = 1e+5 / cost
        else:
            # Infeasible: 懲罰與上次可行 FC 的差距
            reference_FC = 20 * np.log10(self.last_solved_FC)
            FC_dB = 20 * np.log10(FC)
            diff = abs(FC_dB - reference_FC)
            reward = -np.sum(diff * self.parameter.w_earlyTrain)
            sum_error = fc_dist = cost_sum_error = cost_fc_dist = cost_Wgc = 0

        if visual >= 1:
            index = ["sumError", "FCfreq", "Wgc", "status"]
            data = {
                "Value": [sum_error, fc_dist, Wgc, status],
                "Cost": [cost_sum_error, cost_fc_dist, cost_Wgc, reward]
            }
            df = pd.DataFrame(data, index=index)
            print(f"{df}\n")
        
        return reward

class PlotExporter:
    """
    實驗結果繪圖與匯出類別。
    自動在 ExperimentData 下建立以時間命名的資料夾，儲存 Bode 圖框架與 MP4 動畫。
    """
    
    def __init__(self, fps=5):
        """
        初始化匯出器，建立實驗資料夾。
        
        Args:
            fps: MP4 影片幀率
        """
        from datetime import datetime
        
        # 建立時間戳記資料夾
        timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M")
        self.experiment_folder = os.path.join('..', 'ExperimentData', timestamp)
        self.frames_folder = os.path.join(self.experiment_folder, 'frames')
        self.fft_frames_folder = os.path.join(self.experiment_folder, 'fft_frames')
        self.video_name = os.path.join(self.experiment_folder, 'frequency_response.mp4')
        self.fps = fps
        self.step = 0
        self.saved_frames = []

        # 建立資料夾
        os.makedirs(self.frames_folder, exist_ok=True)
        os.makedirs(self.fft_frames_folder, exist_ok=True)
        print(f"實驗資料夾已建立: {self.experiment_folder}")

    def plot_frame(self, CC, plant, FC, manual_add_FC):
        """繪製單張開迴路 Bode 圖 + FC 限制點，並儲存圖片。"""
        plt.figure(figsize=FIG_SIZE_SINGLE)
        OLoop = ctrl.minreal(ctrl.ss2tf(CC * plant), tol=1e-3, verbose=False)
        mag, _, oma = ctrl.bode(OLoop, dB=True, omega_limits=[1e-2, 3e3], plot=False)
        
        # 畫 OLoop
        plt.plot(oma, 20 * np.log10(mag), color='b', label='Open Loop')
        
        # 畫手動添加 FC 點
        if len(manual_add_FC) > 0:
            plt.scatter(manual_add_FC[:, 0], 20 * np.log10(manual_add_FC[:, 1]), 
                       color='g', s=50, zorder=5, label='Manual FC')
        
        # 畫 Actor 產生 FC 點
        plt.scatter(FC[:, 0], 20 * np.log10(FC[:, 1]), 
                   color='r', s=50, zorder=5, label='Actor FC')
        
        # 畫全部 FC 連線
        combined = np.vstack([FC, manual_add_FC]) if len(manual_add_FC) > 0 else FC
        new_FC = combined[np.argsort(combined[:, 0])]
        plt.plot(new_FC[:, 0], 20 * np.log10(new_FC[:, 1]), color='r', linestyle='--', alpha=0.7)

        plt.grid(True)
        plt.xscale('log')
        plt.xlim(1e-2, 1e4)
        plt.ylim(-70, 70)
        plt.xlabel("Frequency (rad/s)")
        plt.ylabel("Magnitude (dB)")
        plt.title(f'Step {self.step}')
        plt.legend(loc='upper right')

        filename = os.path.join(self.frames_folder, f'frame_{self.step:03d}.png')
        plt.savefig(filename, dpi=100)
        plt.close('all')

        self.saved_frames.append(filename)
        self.step += 1

    def plot_error(self, error):
        """繪製追蹤誤差時域圖並儲存。"""
        combined_data = np.concatenate(error)
        rms = np.sqrt(np.mean(combined_data ** 2))
        print(f"RMS Error: {rms:.4f} um")
        
        plt.figure(figsize=FIG_SIZE_WIDE)
        if not np.any(np.abs(combined_data) > 30):
            plt.ylim(-30, 30)
        
        x = np.linspace(0, len(combined_data) * 0.001, len(combined_data))
        plt.plot(x, combined_data)
        plt.title(f"Error (RMS: {rms:.4f} um)", fontsize=18)
        plt.xlabel("Time (s)", fontsize=14)
        plt.ylabel("Magnitude (um)", fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        
        # 儲存圖片
        save_path = os.path.join(self.experiment_folder, 'error.png')
        plt.savefig(save_path, dpi=100)
        plt.show()
        print(f"誤差圖已儲存: {save_path}")

    def save_mp4(self):
        """將所有儲存的圖片製作成 MP4 動畫。"""
        if not self.saved_frames:
            print("沒有可用的圖片框架")
            return
            
        with imageio.get_writer(self.video_name, fps=self.fps, codec='libx264', quality=8) as writer:
            for filename in self.saved_frames:
                image = imageio.imread(filename)
                writer.append_data(image)
        
        print(f"MP4 已儲存: {self.video_name}")
        
        # 重置狀態
        self.saved_frames = []
        self.step = 0

    def save_experiment_info(self, info_dict):
        """儲存實驗資訊到文字檔。"""
        info_path = os.path.join(self.experiment_folder, 'experiment_info.txt')
        with open(info_path, 'w', encoding='utf-8') as f:
            for key, value in info_dict.items():
                f.write(f"{key}: {value}\n")
        print(f"實驗資訊已儲存: {info_path}")

    def get_experiment_folder(self):
        """回傳實驗資料夾路徑。"""
        return self.experiment_folder
