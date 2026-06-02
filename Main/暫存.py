'''路徑微分
def dy_dt(path,Ts):
    time = np.linspace(0, len(path)*Ts, len(path))
    dt = np.diff(time)
    dy = np.diff(path)
    dy_dt = dy / dt
    dy_dt = np.append(dy_dt, dy_dt[-1])#因為微分少一個時間點，所以複製最後一個微分點
    return dy_dt
'''
'''路徑傅立葉轉換
import numpy as np
import scipy
from scipy import fftpack
import matplotlib.pyplot as plt
import CNC

Ts = 0.001
path_model=CNC.PathModel(Ts)
path=path_model.testpath()#取得參考路徑


######全path傅立葉變換########
#N = len(path)
N=100000
yf = scipy.fft.fft(path,N)  # 計算傅立葉轉換
xf = scipy.fft.fftfreq(N, Ts)  # 計算頻率軸

# 只保留正頻部分
real_part = np.real(yf[:N // 2])/N
imag_part = np.imag(yf[:N // 2])/N
magnitude = np.abs(yf[:N // 2])/15000 #頻譜振幅
xf = xf[:N//2]
# 繪製頻譜圖：實部和虛部
plt.figure(figsize=(12, 6))

# 實部折線圖
plt.subplot(3, 1, 1)
plt.plot(xf, real_part, color='blue')
plt.title("FFT Real Part")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.xlim(0, 2.5)
plt.grid()

# 虛部折線圖
plt.subplot(3, 1, 2)
plt.plot(xf, imag_part, color='red')
plt.title("FFT Imaginary Part")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.xlim(0, 2.5)
plt.grid()

# 振幅折線圖
plt.subplot(3, 1, 3)
plt.plot(xf, magnitude, color='green')
plt.title("Path FFT")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 10)
plt.grid()
plt.tight_layout()
plt.show()

#重建完整的頻域信號（包含正頻和負頻）
# 重建完整頻域數據
reconstructed_yf = real_part + 1j * imag_part
# 補充負頻部分，以實現共軛對稱
reconstructed_yf_full = np.concatenate((reconstructed_yf, np.conj(reconstructed_yf[-2:0:-1])))

# 使用 IFFT 還原時域信號
reconstructed_path = np.real(scipy.fft.ifft(reconstructed_yf_full*N))

# 繪製還原後的時域信號
plt.figure(figsize=(10, 4))
plt.plot(reconstructed_path, label="Reconstructed Path")
plt.plot(path, linestyle='dashed', label="Original Path", color='orange')
plt.title("Time Domain Signal Reconstruction")
plt.xlabel("Sample Points")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()


######分段path傅立葉變換########
n=10#第n段
path_district=path[100*(n-1):100*n+1]
#N = len(path_district)
N=50000
yf = scipy.fft.fft(path_district,N)  # 計算傅立葉轉換
xf = scipy.fft.fftfreq(N, Ts)  # 計算頻率軸

# 只保留正頻部分
real_part = np.real(yf[:N // 2])/100
imag_part = np.imag(yf[:N // 2])/100
magnitude = np.abs(yf[:N // 2])/100 #頻譜振幅
xf = xf[:N//2]
#只保留到希望的頻率
#mask=xf<1
#magnitude=magnitude[mask]
#歸一化
magnitude_min = np.min(magnitude)
magnitude_max = np.max(magnitude)
normalized_magnitude= (magnitude - magnitude_min) / (magnitude_max - magnitude_min)

# 繪製頻譜圖：實部和虛部
plt.figure(figsize=(12, 6))
# 實部折線圖
plt.subplot(3, 1, 1)
plt.plot(xf, real_part, color='blue')
plt.title("FFT Real Part")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.xlim(0, 2.5)
plt.grid()

# 虛部折線圖
plt.subplot(3, 1, 2)
plt.plot(xf, imag_part, color='red')
plt.title("FFT Imaginary Part")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.xlim(0, 2.5)
plt.grid()

# 振幅折線圖
plt.subplot(3, 1, 3)
plt.plot(xf, magnitude, color='green')
plt.title("Path district FFT ")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 10)
plt.grid()
plt.tight_layout()
plt.show()

#重建完整的頻域信號（包含正頻和負頻）
# 重建完整頻域數據
reconstructed_yf = real_part + 1j * imag_part
# 補充負頻部分，以實現共軛對稱
reconstructed_yf_full = np.concatenate((reconstructed_yf, np.conj(reconstructed_yf[-2:0:-1])))

# 使用 IFFT 還原時域信號
reconstructed_path = np.real(scipy.fft.ifft(reconstructed_yf_full*N))

# 繪製還原後的時域信號
plt.figure(figsize=(10, 4))
plt.plot(reconstructed_path, label="Reconstructed Path")
plt.plot(path_district, linestyle='dashed', label="Original Path", color='orange')
plt.title("Time Domain Signal Reconstruction")
plt.xlabel("Sample Points")
plt.ylabel("Amplitude")
plt.xlim(0, 100)
plt.legend()
plt.grid()
plt.show()

'''
'''路徑傅立葉使用動態時間遮罩測試
import CNC
import numpy as np
import matplotlib.pyplot as plt
import scipy
import os
import imageio
Ts=0.001
pdl=300
fft_limit_freq=15
path_min_freq=0.2
def segment_path(path, offset=0):#參考路徑切割
    offset=offset%pdl
    path=path[offset:]
    num_path_distric=int(path.shape[0]/pdl)#取整數看路徑能切成多少個片段，最後不完整的放棄
    path_distric = [path[i*pdl : (i+1)*pdl] for i in range(num_path_distric)]#只取路徑部分切割，不包含時間向量
    path_distric.append(np.full(pdl, path_distric[-1][-1]))#為最後一筆next_state創造下100ms的路徑
    return path_distric, num_path_distric
def path_FFT(path, path_index, prev_dominant_freq):
    min_freq=0.2#避免太低頻讓mask長度變無限大
    if prev_dominant_freq<min_freq : prev_dominant_freq=min_freq
    FFT_mask=int(1/prev_dominant_freq/Ts*2)
    N=int(100000/fft_limit_freq)#path_FFT輸出後為100資料點，如果要改成50資料點就改成50000

    if path_index+FFT_mask > len(path):  # 如果不夠 FFT_mask 的長度，取最後滿足 FFT_mask 的片段
        path_mask=path[-FFT_mask:]
    else:
        path_mask=path[path_index : path_index+FFT_mask]

    path_mask=path_mask-np.mean(path_mask)#減掉DC值
    hanning_window = np.hanning(FFT_mask)
    windowed_path=hanning_window*path_mask
    yf = scipy.fft.fft(windowed_path,N)  # 計算傅立葉轉換
    xf = scipy.fft.fftfreq(N, Ts)  # 計算頻率軸

    # 只保留正頻部分
    magnitude = np.abs(yf[:N // 2])/FFT_mask#頻譜振幅
    xf = xf[:N//2]#正頻頻率軸
    #只保留到希望的頻率
    mask=xf<fft_limit_freq
    magnitude=magnitude[mask]
    #歸一化
    magnitude_min = np.min(magnitude)
    magnitude_max = np.max(magnitude)
    normalized_magnitude= (magnitude - magnitude_min) / (magnitude_max - magnitude_min)
    # 找出所有峰值
    peaks, _ = scipy.signal.find_peaks(normalized_magnitude)
    peak_magnitudes = normalized_magnitude[peaks]
    # 篩選符合條件的峰值
    threshold = 0.7  # 設定峰值的門檻
    valid_peaks = peaks[peak_magnitudes >= threshold]
    if len(valid_peaks) > 0:# 找到最小的頻率對應的峰值
        dominant_freq = xf[valid_peaks].min()
    else:# 如果没有符合条件的峰值，则选择最大幅值对应的频率
        dominant_index = np.argmax(normalized_magnitude)
        dominant_freq = xf[dominant_index]
    return normalized_magnitude, dominant_freq, FFT_mask

pathmodel=CNC.PathModel(Ts)

path=pathmodel.up_down_chirp()
path_index=0
num_path_district=int((len(path)-path_index)/pdl)
prev_dominant_freq=path_min_freq

#創建資料夾儲存圖片
if not os.path.exists('frames'):
    os.makedirs('frames')
for step in range(num_path_district):
    FFT_data, prev_dominant_freq, FFT_mask=path_FFT(path, path_index, prev_dominant_freq)
    if step * pdl + FFT_mask > len(path):  # 如果不够 FFT_mask 的长度，取最后满足 FFT_mask 的片段
        path_mask = path[-FFT_mask:]
        mask_start = (len(path) - FFT_mask) * Ts
        mask_end = len(path) * Ts
    else:
        path_mask = path[step * pdl : step * pdl + FFT_mask]
        mask_start = step * pdl * Ts
        mask_end = (step * pdl + FFT_mask) * Ts
    #畫mask在path上
    plt.figure(figsize=(12, 8))
    t=np.arange(0,len(path)*Ts,Ts)
    plt.plot(t,path, zorder=1)
    plt.axvspan(mask_start, mask_end, color='red', alpha=0.3, label="FFT Mask Region", zorder=2)
    plt.xlabel("Time", fontsize=40)
    plt.ylabel("Magnitude", fontsize=40)
    plt.title(f'Dynamic Mask Step {step + 1}', fontsize=40)
    plt.tick_params(axis='both', labelsize=32)
    
    #畫FFT_data
    t=np.arange(0,15,15/len(FFT_data))
    plt.plot(t,FFT_data)
    plt.xlabel("Frequency(Hz)", fontsize=40)
    plt.ylabel("Magnitude", fontsize=40)
    plt.title(f'Path FFT Step {step + 1}', fontsize=40)
    plt.tick_params(axis='both', labelsize=32)    
    
    plt.tight_layout()
    plt.savefig(f'frames/frame_{step:03d}.png')
    plt.clf()  # 清空當前圖表
    path_index=path_index+pdl
with imageio.get_writer('animation.mp4', fps=5, codec='libx264', quality=8) as writer:
    for i in range(num_path_district):
        image = imageio.v2.imread(f'frames/frame_{i:03d}.png')
        writer.append_data(image)
'''
'''切換控制器誤差觀察
import numpy as np
import CNC
import argparse
import matplotlib.pyplot as plt

np.set_printoptions(precision=5,suppress=True)#設置打印位數，科學記號

######   參數區域    ######
#region
#CNC參數
x_polegain=0.4352
z_polegain=0.4952
parser2 = argparse.ArgumentParser(description="CNC參數")
parser2.add_argument('--Lq', type=int, default=10)
parser2.add_argument('--gcfUpper', type=int, default=300)#x=x軸
parser2.add_argument('--gcfLower', type=int, default=300-10)
parser2.add_argument('--damp', type=float, default=0.0)
parser2.add_argument('--w_gcfOutside', type=float, default=1e+7)#w=權重
parser2.add_argument('--w_gcfInside', type=float, default=1e+5)
parser2.add_argument('--w_damp', type=float, default=5e+7)
parser2.add_argument('--w_sumError', type=float, default=2e+6)
parser2.add_argument('--w_FCfreq', type=float, default=5e+3)
parser2.add_argument('--w_FCgain', type=float, default=1e+7)
parser2.add_argument('--w_earlyTrain', type=float, default=5e-3)
parser2.add_argument('--w_fftErr_Residual', type=float, default=2e+7)
CNC_parameter = parser2.parse_args()
#其他參數
visual=0
Ts=0.001
path_min_freq=0.5#chipsin最小頻率
path_max_freq=1#chipsin最大頻率
pdl=100#path_distric_len 多少ms一個區間
FFT_limit_freq=2#path_FFT取到幾HZ
FC = np.array([
    [0.1, 1000],
    [1, 100],
    [10, 10],
    [100, 1.1],
    [300, 0.8],
    [400, 0.56],
    [500, 0.39],
    [600, 0.26],
    [700, 0.16],
    [800, 0.09],
    [900, 0.07],
    [1000, 0.08],
    [1300, 0.1],        
    [1500, 0.09],
    [1800, 0.06],
    [2000, 0.03],
    [2500, 0.07],
    [3000, 0.1]
])
FC2 = np.array([
[   0.11782 ,2001.10516],
 [   2.49594,  123.86329],
 [   9.55501,   13.21885],
 [ 111.43296 ,   1.11843],
 [1172.57396 ,   0.2936 ],
 [1737.00788,    0.49927],
 [2308.91069,    0.20494],
 [1565.82957,    0.13543],
 [2140.35176,    0.05406],
 [2472.43871 ,   0.01239],
 [2031.91877 ,   0.04231],
 [1916.4614 ,    0.05511],
 [2599.10781,    0.10984],
 [2235.38072 ,   0.00918],
 [2518.89033 ,   0.02819],
 [2522.13596,    0.00553],
 [2218.52915 ,   0.06772],
 [2474.59833,    0.01594]
 ])
#endregion

model_x = CNC.CNCModel('x',Ts)#創建馬達實例
ID_Plant=model_x.ID_Plant()#取得馬達原始模型
pathmodel=CNC.PathModel(Ts)
allpath=pathmodel.training_path()
path=pathmodel.test_path()
costfunction_x=CNC.Costfunction(CNC_parameter, x_polegain, ID_Plant, path, pdl, FC2.copy())

error=[]
path_index=0
num_path_district=int((len(path)-path_index)/pdl)
for step in range(num_path_district):
    status, CC, ek_hat=costfunction_x.switch_controller(path, path_index, FC.copy())
    error.append(ek_hat)
    path_index=path_index+pdl
combined_data = np.concatenate(error)
print(np.sqrt(np.mean(combined_data**2)))
x = np.linspace(0, len(combined_data)*0.001, len(combined_data))
plt.plot(x,combined_data)
plt.title("Error")
plt.xlabel("time(s)")
plt.ylabel("Magnitude(um)")
plt.grid()
plt.show()
'''
'''路徑多項式擬和
import numpy as np
import matplotlib.pyplot as plt
import CNC

Ts = 0.001
path_model=CNC.PathModel(Ts)
path=path_model.testpath()#取得參考路徑

degree = 5  # 選擇擬合多項式的階數
######全path多項式擬和########
t=np.arange(0,len(path)*Ts,Ts)

# 使用 polyfit 擬合多項式
coefficients = np.polyfit(t, path, degree)

# 生成多項式
polynomial = np.poly1d(coefficients)

# 使用多項式生成擬合的路徑
fitted_path = polynomial(t)

# 可視化原始數據和擬合後的路徑
plt.plot(t, path, label="Original Path", color="blue")
plt.plot(t, fitted_path, label=f"Polynomial Fit (Degree {degree})", color="red", linestyle="--")
plt.xlabel("Time(s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()

######分段path多項式擬和########
n=142#第n段
path_district=path[100*(n-1):100*n]
t=np.arange(0,len(path_district)*Ts,Ts)

# 使用 polyfit 擬合多項式
coefficients = np.polyfit(t, path_district, degree)

# 生成多項式
polynomial = np.poly1d(coefficients)

# 使用多項式生成擬合的路徑
fitted_path = polynomial(t)

# 可視化原始數據和擬合後的路徑
plt.plot(t, path_district, label="Original Path", color="blue")
plt.plot(t, fitted_path, label=f"Polynomial Fit (Degree {degree})", color="red", linestyle="--")
plt.xlabel("Time(s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()
'''
'''畫ID model Uncertainty model波德圖
import matplotlib.pyplot as plt
import control as ctrl
import numpy as np
import CNC
model_x = CNC.CNCModel('x',0.001)#創建馬達實例
ID_Plant=model_x.ID_Plant()#取得馬達ID模型
uncertainty_Plant=model_x.uncertainty_Plant()#取得馬達不確定模型

# 定义频率范围
omega = np.logspace(np.log10(0.1), np.log10(3000), num=5000)   # 频率范围从0.1到3000 rad/s

# 大小波德圖
plt.figure(figsize=(8, 6))

mag1, phase1, omega1 = ctrl.bode(ID_Plant['v2p'], omega, dB=True, plot=False)
plt.subplot(2, 1, 1)
plt.semilogx(omega1, 20 * np.log10(mag1))  # 将频率转换为 Hz
plt.title('ID model')
plt.xlabel('Frequency [rad/s]')
plt.ylabel('Magnitude [dB]')
plt.grid(True, which='both', linestyle='--')

mag2, phase2, omega2 = ctrl.bode(uncertainty_Plant['v2p'], omega, dB=True, plot=False)
plt.subplot(2, 1, 2)
plt.semilogx(omega2, 20 * np.log10(mag2))  # 将频率转换为 Hz
plt.title('Uncertainty model')
plt.xlabel('Frequency [rad/s]')
plt.ylabel('Magnitude [dB]')
plt.grid(True, which='both', linestyle='--')

plt.subplots_adjust(hspace=0.7)
plt.show()

# 相位波德圖
plt.figure(figsize=(8, 6))

plt.subplot(2, 1, 1)
plt.semilogx(omega1 , phase1 * (180 / np.pi))
plt.title('ID model')
plt.xlabel('Frequency [rad/s]')
plt.ylabel('Phase [degrees]')
plt.grid(True, which='both', linestyle='--')

plt.subplot(2, 1, 2)
plt.semilogx(omega2, phase2 * (180 / np.pi))
plt.title('Uncertainty model')
plt.xlabel('Frequency [rad/s]')
plt.ylabel('Phase [degrees]')
plt.grid(True, which='both', linestyle='--')

plt.subplots_adjust(hspace=0.7)
plt.show()
'''
'''畫隨機共振峰值(加上控制器)
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
'''畫隨機共振峰值(純受控體)
import CNC
import numpy as np
import control as ctrl
import matplotlib.pyplot as plt
Ts=0.001
model_x = CNC.CNCModel('x',Ts)#創建馬達實例
plt.figure(figsize=(12, 8))
plant=model_x.ID_Plant()
mag, _, oma = ctrl.bode(plant['v2v'], dB=True, omega_limits=[1e-2, 3e3], plot=False)
plt.plot(oma, 20 * np.log10(mag), color='b', linewidth=2)
#CNC中uncertainty_Plant要改v2v
for i in range(100):
    Plant=model_x.uncertainty_Plant()
    mag, _, oma = ctrl.bode(Plant['v2p'], dB=True, omega_limits=[1e-2, 3e3], plot=False)
    plt.plot(oma, 20 * np.log10(mag), color='r', linewidth=0.3)
plt.grid()
plt.xscale('log')
plt.xlim(1, 1e4)
plt.ylim(-70, 70)
plt.xlabel("Frequency (rad/s)", size=28)
plt.ylabel("Magnitude (dB)", size=28)
plt.title("Baseline Uncertainty Ensemble", size=36)
plt.tick_params(axis='both', labelsize=20)
plt.show()
'''
'''畫系統鑑別輸入
import scipy
import numpy as np
import matplotlib.pyplot as plt

Ts = 0.001
path_time = 5
Magnitude=100
t = np.arange(0, path_time, Ts)

inputdata=scipy.signal.chirp(t, f0=0, f1=50, t1=path_time, method='linear',phi=-90)*Magnitude

path=inputdata
inputdata_plot = np.column_stack((t, path))
plt.figure(figsize=(10, 4))
plt.plot(inputdata_plot[:, 0], inputdata_plot[:, 1])
plt.title('Chirp')
plt.xlabel('Time (s)')
plt.ylabel('Magnitude(rpm)')
plt.grid(True)
plt.show() 
'''
'''轉出CC
        CC_tf = ctrl.ss2tf(CC)
        den = np.array(CC_tf.den[0][0], dtype=np.float32)
        cdl=len(den)#controller_data_len
        num = np.array(CC_tf.num[0][0], dtype=np.float32)
        num = np.pad(num, (0, cdl - len(num)), mode='constant')#補齊避免分子階數不足
'''
'''畫神經網路輸出
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
'''畫參考路徑
import numpy as np
import matplotlib.pyplot as plt
import CNC

path_model=CNC.PathModel(0.001)
path_model.plot_path()
'''
'''隨機共振峰值上下界
import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
import CNC

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 參數設定
Ts = 0.001
axis = 'x'

# 建立CNC模型
cnc_model = CNC.CNCModel(axis, Ts)

# 取得ID_Plant (基礎模型)
ID_plant = cnc_model.ID_Plant()
v2v_ID = ID_plant['v2v']

# 設定頻率範圍 (最高到奈奎斯特頻率)
omega_nyquist = 500 * 2 * np.pi  # 奈奎斯特頻率 = 500 Hz = 3141.6 rad/s
omega_range = np.logspace(0, np.log10(omega_nyquist), 1000)  # 1 ~ 3141.6 rad/s

# 計算 ID_Plant 的頻率響應
mag_ID, phase_ID, omega_out = ctrl.frequency_response(v2v_ID, omega_range)

# 轉換為 dB
mag_ID_dB = 20 * np.log10(np.abs(mag_ID.flatten()))

# 計算上界和下界曲線 (只在 300-1000 rad/s 範圍內)
# 共振轉移函數在共振頻率處的增益 = gain / 2
# gain_min = (5 * x * exp(2x) + 2) / 2
# gain_max = (10 * x * exp(2x) + 2) / 2
# 其中 x = (omega - 300) / (1000 - 300)
min_resonance_omega = 300
max_resonance_omega = 1000

omega_flat = omega_out.flatten()

# 只取 300-1000 rad/s 範圍內的頻率
mask = (omega_flat >= min_resonance_omega) & (omega_flat <= max_resonance_omega)
omega_bounds = omega_flat[mask]
mag_ID_bounds = mag_ID_dB[mask]

gain_lower = np.ones_like(omega_bounds)
gain_upper = np.ones_like(omega_bounds)

for i, omega in enumerate(omega_bounds):
    x = (omega - min_resonance_omega) / (max_resonance_omega - min_resonance_omega)
    gain_lower[i] = (5 * x * np.exp(2*x) + 2) / 2
    gain_upper[i] = (10 * x * np.exp(2*x) + 2) / 2

# 上下界 = ID_Plant增益 + 共振增益
gain_lower_dB = mag_ID_bounds + 20 * np.log10(gain_lower)
gain_upper_dB = mag_ID_bounds + 20 * np.log10(gain_upper)

# 繪製波德圖
plt.figure(figsize=(6, 4))

plt.semilogx(omega_out.flatten(), mag_ID_dB, 'b-', linewidth=1.5, label='Nominal Plant')
plt.semilogx(omega_bounds, gain_upper_dB, 'r:', linewidth=2, label='Upper Bound')
plt.semilogx(omega_bounds, gain_lower_dB, 'g--', linewidth=1.5, label='Lower Bound')

# 添加垂直線標記 300 和 1000 rad/s
plt.axvline(x=300, color='k', linestyle='--', linewidth=1, alpha=0.7)
plt.axvline(x=1000, color='k', linestyle='--', linewidth=1, alpha=0.7)

plt.xlabel('Frequency (rad/s)', fontsize=12)
plt.ylabel('Magnitude (dB)', fontsize=12)
plt.title('Random Resonance Peak Bounds', fontsize=14)
plt.legend(loc='best', fontsize=10)
plt.grid(True, which='both', alpha=0.3)
plt.xlim([100, omega_nyquist])
plt.ylim([-40, 40])

# 在 X 軸上添加 300 和 1000 的刻度
ax = plt.gca()
xticks = list(ax.get_xticks())
xticks.extend([300, 1000])
ax.set_xticks([100, 300, 1000, omega_nyquist])
ax.set_xticklabels(['100', '300', '1000', f'{omega_nyquist:.0f}'])

plt.tight_layout()
plt.savefig('Bode_Plot_new.png', dpi=150)
plt.show()
'''