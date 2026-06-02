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

def SimulateResponse(path, CC, plant, X0=0, Ts=0.001):
    plant=ctrl.tf2ss(plant)
    OLoop=ctrl.ss2tf(CC * plant)
    num=OLoop.num[0][0]
    den=OLoop.den[0][0]
    if  len(num) < len(den):# 檢查OLoop分子分母長度
        num = np.pad(num, ( len(den) - len(num), 0), 'constant')
    CLoop=ctrl.TransferFunction(num,den+num,Ts)

    T = np.arange(0, len(path)*Ts, Ts)  # 從0開始到total_time結束，步長為0.001秒
    U=path
    CLoop=ctrl.tf2ss(CLoop)
    _, y , X= ctrl.forced_response( CLoop, T, U, X0, return_x=True)
    X0=CLoop.A@(X[:,-1].reshape(-1,1))+CLoop.B*U[-1]#拿到101時間步的狀態
    ek=(U-y)*1000#誤差，單位um
    return X0, ek, y

def find_Wgc(CC, plant, Ts): 
    OLoop=ctrl.minreal( ctrl.ss2tf(CC * plant), tol=1e-3, verbose=False)

    num=OLoop.num[0][0]
    den=OLoop.den[0][0]
    if  len(num) < len(den):# 檢查OLoop分子分母長度
        num = np.pad(num, ( len(den) - len(num), 0), 'constant')
    CLoop=ctrl.TransferFunction(num,den+num,Ts)
    #_,_,_=ctrl.bode(OLoop, dB=True, omega_limits=[1e-2, 3e3], plot=True)#OLoop波德圖
    #plt.show()

    # 計算PM跟Wgc增益交越頻率
    #region
    b, a = OLoop.num, OLoop.den
    a=a[0][0]
    b=b[0][0]
    n = len(a) - 1  # 系統的階數
    if len(b) < n+1:#把分子長度補到跟分母一樣
        padding = n+1 - len(b)
        b = np.pad(b, (padding, 0), 'constant')
    # 計算第一類切比雪夫多項式的係數
    T = np.zeros((n+1, n+1))
    if n == 1:
        T[:, 0], T[:, 1] = [0, 1], [1, 0]
    else:
        T[n, 0], T[n-1, 1] = 1, 1
        for i in range(2, n+1):  # 在 Python 中範圍是從 2 到 n
            T[:, i] = np.append(2 * T[1:n+1, i-1], 0) - T[:, i-2]
                
    P1, P2 = np.zeros(n+1), np.zeros(n+1)
    for i in range(n+1):
        for k in range(n+1):
            P1 += a[i] * a[k] * T[:, abs(k-i)]      #a*real(rho*rho')*a';  rho = [1;e^{-jw};...;e^{-jnw}]
            P2 += b[i] * b[k] * T[:, abs(k-i)]     #b*real(rho*rho')*b'
            
    R=P1-P2
    x = np.roots(R)
    x = x[np.isreal(x)]
    x = np.real(x[np.abs(x) < 1])
    max_index = np.argmax(x)
    w = np.arccos(x[max_index]).item()
    Wgc=w/Ts#從0~pi轉成角頻率
    #endregion

    return Wgc

def find_resonance(CC, plant, Ts, path_district, ek):
    def FT(data, Ts):
        # 先經過 Hanning 窗避免邊界效應
        pdl=len(data)
        hanning_window = np.hanning(pdl)
        windowed_data = hanning_window * data
        # 傅立葉轉換
        N = int(1/Ts)
        yf = scipy.fft.fft(windowed_data, N)  # 計算傅立葉轉換
        xf = scipy.fft.fftfreq(N, Ts)  # 計算頻率軸
        # 只保留正頻部分
        magnitude = np.abs(yf[:N // 2]) / pdl  # 頻譜振幅
        xf = xf[:N // 2]  # 正頻頻率軸
        return xf, magnitude
    Wgc=find_Wgc(CC, plant, Ts)
    Fgc=Wgc/(2*np.pi)
    output=path_district-ek
    ek=ek-ek[0]
    #轉傅立葉
    eFTx, eFTy=FT(ek, Ts)
    _, oFTy=FT(output,Ts)
    #找到需要處理的共振頻率跟震幅
    peaks_position, _ = scipy.signal.find_peaks(eFTy)
    mask = (eFTx[peaks_position] > Fgc) & (eFTy[peaks_position] > np.mean(oFTy)*5)
    HFRw=(eFTx[peaks_position][mask])*2*np.pi#共振頻率
    HFRm=(eFTy[peaks_position][mask])*2*np.pi#共振震幅
    eMag=eFTy[peaks_position][mask]
    oMag=oFTy[peaks_position][mask]
    HFRg=20*np.log10(eMag/oMag)#共振增益dB
    '''
    plt.plot(eFTy)
    plt.axhline(y=np.mean(oFTy)*2, color="r")
    plt.axvline(x=Fgc, color="g")
    plt.xlabel("Freq(Hz)")
    plt.ylabel("Mag")
    plt.title("Find resonance")
    plt.show()
    #'''
    return HFRw, HFRm, HFRg

class CNCModel:
    def __init__(self,axis,Ts):
        self.Ts = Ts
        self.integrator = ctrl.TransferFunction([Ts, 0], [1, -1], Ts)#積分器
        self.axis=axis

    def ID_Plant(self):
        # X軸馬達轉移函數
        num_x = [0,	0.0410789388950551,	0.116567016168533]
        den_x = [1,	-1.41353788543924,	0.566877911191563]
        Pv_x = ctrl.TransferFunction(num_x, den_x, self.Ts)    # 創建X軸轉移函數模型
        rpm2mms_x = 10/60       # 轉動速度轉為每秒多少mm 
        Px = Pv_x * rpm2mms_x * self.integrator# 輸入轉速輸出位置轉移函數。把轉速轉為x軸速度並積分
        
        # Z軸馬達轉移函數(廢棄)
        num_z = [0,0.0973217504156460,-0.209580502231482,0.151454774135651,-0.0370139184925609]
        den_z = [1,-3.44700230387150,4.55138855100522,-2.73113485001183,0.628907804339994]
        Pv_z = ctrl.TransferFunction(num_z, den_z, self.Ts)
        rpm2mms_z = 12/60  
        Pz = Pv_z* rpm2mms_z * self.integrator
        v2v={}
        v2v['x']=Pv_x
        v2v['z']=Pv_z
        v2p={}
        v2p['x'] = Px
        v2p['z'] = Pz

        plant = {
            'v2v' : v2v[self.axis],
            'v2p': v2p[self.axis],
            'Ts': self.Ts
        }
        return plant

    def test_Plant(self):
        # X軸馬達轉移函數
        ID_model=self.ID_Plant()
        v2p=ID_model['v2p']

        #omega400大小 3.8
        #omega600  7.5

        omega=600
        zeta = 0.01  # 阻尼比
        resonance_tf_continuous = ctrl.TransferFunction([1,  12* zeta * omega, omega**2], [1, 2 * zeta * omega, omega**2])# 共振點的二階系統
        resonance_tf = ctrl.sample_system(resonance_tf_continuous, self.Ts)
        # 高精度打印
        # np.set_printoptions(precision=15, suppress=True)
        # print("resonance_tf 分子:", resonance_tf.num[0][0])
        # print("resonance_tf 分母:", resonance_tf.den[0][0])
        v2p= v2p * resonance_tf# 更新轉移函數

        plant = {
            'v2p': v2p,
            'Ts': self.Ts
        }
        return plant

    def uncertainty_Plant(self):
        def cancel_pole_zero(z, p, tol=1e-3):
            """自動進行 pole-zero 對消，回傳已對消的新 z/p"""
            z_new = list(z)
            p_new = list(p)

            for zero in z:
                # 在 pole 裡找最接近的
                diffs = [abs(zero - pole) for pole in p_new]
                if len(diffs) == 0:
                    continue
                min_diff = min(diffs)
                if min_diff < tol:
                    idx = diffs.index(min_diff)
                    # 移除對消的 pole 和 zero
                    z_new.remove(zero)
                    del p_new[idx]
            return np.array(z_new, dtype=complex), np.array(p_new, dtype=complex)
        #ID_Plant 的zpk
        ID_Plant=self.ID_Plant()
        num = ID_Plant["v2p"].num[0][0]
        den = ID_Plant["v2p"].den[0][0]
        IDz, IDp, IDk = tf2zpk(num, den)

        # 載入 .mat 檔
        data = scipy.io.loadmat('Delta_Data.mat')
        z_all = data['z_all'].squeeze()
        p_all = data['p_all'].squeeze()
        k_all = data['k_all'].squeeze()
        Ts = float(data['Ts'])

        #合成uncertainty model
        v2ps = []
        for i in range (0,len(z_all)):
            deltaz = z_all[i].squeeze().astype(complex)
            deltap = p_all[i].squeeze().astype(complex)
            deltak = k_all[i].item()
            #組合
            z = np.concatenate([deltaz, IDz])
            p = np.concatenate([deltap, IDp])
            k=deltak*IDk
            z_new, p_new = cancel_pole_zero(z, p, tol=1e-2)
            v2ps.append(ctrl.zpk(z_new, p_new, k, Ts))
        
        #隨機選擇
        i = np.random.randint(0, len(z_all))
        v2p=v2ps[i]

        plant = {
            'v2p': v2p,
            'Ts': self.Ts
        }
        return plant

    def random_Plant(self, min_resonance_omega=300, max_resonance_omega=1000):
        uncertainty_Plant=self.uncertainty_Plant()
        v2p=uncertainty_Plant['v2p']

        #隨機頻率
        omega_candidates = np.linspace(min_resonance_omega, max_resonance_omega, num=500)        # 建立頻率候選點
        alpha = np.log(5) / (max_resonance_omega - min_resonance_omega)
        weights = np.exp(-alpha * (omega_candidates - min_resonance_omega))# 頻率權重設計：指數遞減，讓最低頻率出現的機率為最高的 n 倍
        weights /= np.sum(weights)  # 正規化成機率分布
        omega = np.random.choice(omega_candidates, p=weights)# 根據權重抽一個頻率
        #隨機阻尼
        zeta = np.random.uniform(0.005, 0.05)  # 阻尼比 (0.005~0.05)

        #隨機峰值大小
        norm_freq = (omega - min_resonance_omega) / (max_resonance_omega - min_resonance_omega)# 歸一化頻率（低頻：0，高頻：1）
        gain_min = 5 * norm_freq * np.exp(norm_freq*2) +2
        gain_max = 10 * norm_freq *  np.exp(norm_freq*2) +2
        gain = np.random.uniform(gain_min, gain_max)# 依照頻率選取隨機峰值
        #添加隨機共振點
        resonance_tf_continuous = ctrl.TransferFunction([1, gain * zeta * omega, omega**2], [1, 2 * zeta * omega, omega**2])# 共振點的二階系統
        resonance_tf = ctrl.sample_system(resonance_tf_continuous, self.Ts)
        v2p= v2p * resonance_tf# 更新轉移函數

        plant = {
            'v2p': v2p,
            'Ts': self.Ts
        }
        return plant

class PathModel:
    def __init__(self, Ts, path_time=15):
        self.Ts = Ts
        self.path_time=path_time

    def noise_path(self):
        #在中間加入噪聲
        path_time=self.path_time
        Ts=self.Ts
        t = np.arange(0, path_time, Ts)
        #基波
        ref =self.test_path()
        #第一雜訊
        A_noise1=0.2        #震幅
        f_noise1=127    #頻率Hz
        t_on1=5     #進入時間
        t_off1=5.2
        noise1= A_noise1 * np.sin(2 * np.pi * f_noise1 * t) * (t >= t_on1)*(t <= t_off1)
        #混頻
        ref+= noise1
        return ref

    def test_path(self):#0~1Hz
        Ts = self.Ts  # 采樣時間
        path_time = self.path_time
        Magnitude=1
        t = np.arange(0, path_time, Ts)

        inputdata=scipy.signal.chirp(t, f0=0, f1=1, t1=path_time, method='linear',phi=-90)*Magnitude
        return inputdata

    def test_path2(self):#0~8Hz
        Ts = self.Ts  # 采樣時間
        path_time = self.path_time
        Magnitude=1
        t = np.arange(0, path_time, Ts)

        inputdata=scipy.signal.chirp(t, f0=0, f1=8, t1=path_time, method='linear',phi=-90)*Magnitude
        return inputdata

    def training_path(self):
        Ts = self.Ts
        path_time = self.path_time
        t = np.arange(0, path_time, Ts)

        paths = [
            0.55*np.cos(2*np.pi*0.1*t+np.pi/9) + 0.35*np.sin(2*np.pi*0.3*t+np.pi/5) + 0.1,
            0.6*np.cos(2*np.pi*0.4*t+np.pi/7) + 0.25*np.cos(2*np.pi*0.8*t+np.pi/3) - 0.1,
            0.4*np.sin(2*np.pi*1.0*t+np.pi/4) + 0.35*np.cos(2*np.pi*1.8*t+np.pi/6) + 0.25*np.sin(2*np.pi*0.6*t+np.pi/2),
            0.5*np.cos(2*np.pi*3.0*t+np.pi/2) + 0.4*np.sin(2*np.pi*5.5*t+np.pi/9) - 0.15,
            0.45*np.sin(2*np.pi*0.2*t+np.pi/3) + 0.4*np.cos(2*np.pi*6.0*t+np.pi/10) + 0.15,
            0.7*np.cos(2*np.pi*0.05*t+np.pi/8) + 0.2,
            0.35*np.sin(2*np.pi*1.0*t+np.pi/5) + 0.35*np.sin(2*np.pi*1.3*t+np.pi/7) + 0.25*np.cos(2*np.pi*1.6*t+np.pi/2),
            0.5*np.cos(2*np.pi*2.5*t+np.pi/6) + 0.3*np.cos(2*np.pi*5.0*t+np.pi/4) - 0.1,
            0.4*np.sin(2*np.pi*7.0*t+np.pi/3) + 0.3*np.cos(2*np.pi*9.0*t+np.pi/10),
            0.6*np.sin(2*np.pi*4.5*t+np.pi/2) + 0.2,
            0.8*scipy.signal.chirp(t, f0=0.02, f1=0.2,  t1=path_time, method='linear', phi=-40) + 0.05,  # 超低頻
            0.8*scipy.signal.chirp(t, f0=0.1,  f1=0.8,  t1=path_time, method='linear', phi=-70) - 0.05,  # 低頻
            0.8*scipy.signal.chirp(t, f0=0.3,  f1=1.5,  t1=path_time, method='linear', phi=-20),         # 低～中
            0.8*scipy.signal.chirp(t, f0=0.8,  f1=3.0,  t1=path_time, method='linear', phi=-90) + 0.05,  # 中
            0.8*scipy.signal.chirp(t, f0=2.0,  f1=4.0,  t1=path_time, method='linear', phi=-10),         # 中
            0.8*scipy.signal.chirp(t, f0=3.0,  f1=7.0,  t1=path_time, method='linear', phi=-60) - 0.05,  # 中～高
            0.8*scipy.signal.chirp(t, f0=0.0,  f1=9.0,  t1=path_time, method='linear', phi=-110) + 0.05, # 高
            0.8*scipy.signal.chirp(t, f0=0.5,  f1=0.6,  t1=path_time, method='linear', phi=-30),         # 低頻窄帶
            0.8*scipy.signal.chirp(t, f0=1.5,  f1=0.2,  t1=path_time, method='linear', phi=-50) + 0.05,  # 反向低～中
            0.8*scipy.signal.chirp(t, f0=9.0,  f1=3.0,  t1=path_time, method='linear', phi=-75) - 0.05,  # 反向高→中
        ]
        return paths

    def up_down_chirp(self):
        path_time=self.path_time*2
        Ts=self.Ts
        t = np.arange(0, path_time, Ts)
        fmin = 0   # 0Hz附近(+-1) 你可改成 0.2~1.0
        fmax = 8   # 8Hz附近(+-1) 你可改成 7.0~9.0
        A = 1.0      # 振幅
        phi0 = 0
        offset = 0.0

        phi = 2*np.pi*(fmin*t + (fmax-fmin)*(0.5*t - (path_time/(4*np.pi))*np.sin(2*np.pi*t/path_time))) + phi0
        path_updown_chirp = A*np.sin(phi) + offset
        return path_updown_chirp

    def plot_path(self, pathType=0):
        t = np.arange(0, self.path_time, self.Ts)
        if pathType == 0:
            allpath = self.training_path()
            n_paths = len(allpath)
            # 自動決定子圖排版：接近正方形
            cols = int(np.ceil(np.sqrt(n_paths)))
            rows = int(np.ceil(n_paths / cols))
            # 圖的大小也跟 rows / cols 成比例調整
            plt.figure(figsize=(4 * cols, 3 * rows))
            plt.suptitle('Training Paths', fontsize=24)
            for i, path in enumerate(allpath):
                ax = plt.subplot(rows, cols, i + 1)
                ax.plot(t, path)
                ax.grid(True)
                ax.set_title(f"Path {i+1}", fontsize=18)
                ax.tick_params(axis='both', labelsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # 預留空間給 suptitle
            plt.show()
        if pathType==1:
            path=self.test_path()
            #path=self.up_down_chirp()
            inputdata_plot = np.column_stack((t, path))
            plt.figure(figsize=(10, 4))
            plt.plot(inputdata_plot[:, 0], inputdata_plot[:, 1])
            plt.title('Path', fontsize=18)
            plt.xlabel('Time (s)', fontsize=14)
            plt.ylabel('Magnitude(mm)', fontsize=14)
            plt.grid(True)
            plt.show() 

    def output_path(self, path):
        #'''輸出成txt
        np.savetxt("path.txt", path, delimiter=',', fmt='%.6f')
        #'''
        '''輸出成excel
        df = pd.DataFrame(path)
        df.to_excel('output.xlsx', index=False, engine='openpyxl')
        #'''

class Costfunction:
    def __init__(self, CNC_parameter, polegain, plant, path, pdl, numFC, numLowFreq, manual_FC=-1):
        def coprime_factorization_ss(plant, polegain):#互質分解
            Ts=plant['Ts']
            poles=ctrl.pole(plant['v2p'])
            G=ctrl.tf2ss(plant['v2p'])#原始轉移函數
            #重新指定極點位置
            adjusted_poles =[]
            for pole in poles: 
                if abs(pole) > 0.99:
                    adjusted_poles.append(pole * polegain)
                else:
                    adjusted_poles.append(pole)
            F = ctrl.place(G.A, -1*G.B, adjusted_poles);   #回授增益  極點置換法
            H = ctrl.place(G.A.T, -1*G.C.T, adjusted_poles);  #觀測器增益  極點置換法
            # 根據 F 和 H 計算互質分解的 M, N, X, W
            M = ctrl.ss(G.A + G.B @ F, G.B, F, 1, Ts)
            N = ctrl.ss(G.A + G.B @ F, G.B, G.C + G.D @ F, G.D, Ts)
            X = ctrl.ss(G.A + H.T @ G.C, H.T, F, np.zeros_like(G.D), Ts)
            W = ctrl.ss(G.A + H.T @ G.C, -G.B - H.T @ G.D, F, np.ones_like(G.D), Ts)

            # 更新 plant 並返回
            plant['G']=G
            plant['F']=F
            plant['H']=H
            plant['assignedpole'] = adjusted_poles
            plant['M'] = M
            plant['N'] = N
            plant['X'] = X
            plant['W'] = W
            return plant

        def linear_fractional_transformation(plant):#線性分式轉換
            G =plant['G']
            F =  plant['F']
            H = plant['H']
            L = H.T 
            JA = G.A + G.B @ F + L @ G.C + L @ G.D @ F
            JB = np.concatenate((L, G.B + L @ G.D), axis=1)
            JC = np.concatenate((F, (G.C + G.D @ F)), axis=0)
            n=G.D.shape[0] # 獲取 G.D 矩陣的尺寸
            I = np.eye(n) # 創建單位矩陣
            JD = np.vstack([np.hstack([np.zeros((n, n)), I]), np.hstack([I,  G.D])])#vstake為堆疊矩陣
        
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
                status, _=self.optimizationcvx(path, 0, FC)
                if status=="Solved":
                    self.last_solved_FC=FC.copy()
                    break
                else:
                    scalMag=scalMag*0.9

        self.parameter=CNC_parameter
        self.plant=linear_fractional_transformation(coprime_factorization_ss(plant, polegain))
        self.path=path
        self.pdl=pdl
        self.numLowFreq=numLowFreq
        self.resonanceTable=np.zeros((3,1))

        #Gurobi解QCQP
        self.model=Model("QCQP")
        self.model.setParam('OutputFlag', 0)
         
        #Controller跟last_solved_FC
        if manual_FC!=-1:
            self.set_controller(manual_FC)
        else:
            use_central_controller()
         
        #先初始化一次
        self.initialize()
         
    def initialize(self):#每一輪一開始初始化
         self.X0_hat=0
         self.resonanceTable=np.zeros((5,1))

    def set_controller(self, FC):
        state, Q=self.optimizationcvx(self.path, 0, FC.copy())
        if state=="Solved":
            self.last_solved_FC=FC
            self.CC=self.LFTExpandedSS(Q)
        else:
            self.last_solved_FC=FC
            self.CC=self.LFTExpandedSS(np.zeros((self.parameter.Lq, 1)))
            print("Manual FC無解，使用Central Controller")
        
    def optimizationcvx(self, path, path_index, FC):
        def Time_domain_coefficient(path, path_index):#QCQP的時域係數
            Ts=self.plant['Ts']
            M=self.plant['M']
            N=self.plant['N'] 
            W=self.plant['W']
            L=self.parameter.Lq

            #得到path經過各個狀態矩陣後得到V=NMR，Z=WMR  
            path_=path[path_index : path_index+self.pdl]#
            ym, _,_ =lsim(M, U=path_, T=np.arange(len(path_))*Ts)   #輸出Tm=T=np.arange(len(path))*Ts複製而已
            zk,_ , _ = lsim(W, U=ym, T=np.arange(len(path_))*Ts)
            vk,_, _ = lsim(N, U=ym, T=np.arange(len(path_))*Ts)
         
            #初始化各項係數
            Tphi = np.zeros((L,1))
            Tcoe2_t = np.zeros((L, L))
            Tcoe1_t = np.zeros((1, L))
            Tcoe0_t = 0
        
            #計算係數
            for k in range(len(vk)):
                #wk = 1 + alpha * lambda_**(k)
                wk=0.7**(k)
                Tphi = np.vstack((vk[k], Tphi[:-1]))
                Tcoe2_t += wk * np.outer(Tphi, Tphi)  # np.outer() 是向量外積，出來是10*10矩陣
                Tcoe1_t += wk * zk[k] * Tphi.T    #1*10矩陣
                Tcoe0_t += wk * zk[k]**2    #常量

            return Tcoe2_t, Tcoe1_t,Tcoe0_t

        def Frequence_domain_coefficient(FC):#QCQP的頻域係數
            L=self.parameter.Lq
            Nc = FC.shape[0]#限制條件個數
            alpha = np.ones(Nc)
            beta = np.zeros((L, Nc))
            Omega = np.zeros((L, L, Nc))
            gamma = np.zeros(Nc)
            weight = np.zeros(Nc)
            w0=100000
            N = self.plant['N'];
            M = self.plant['M'];
            X = self.plant['X'];
            W = self.plant['W'];
            P = self.plant['v2p'];
            Ts = self.plant['Ts'];

            # 檢查 FC（只避開 ki=1；大於1往下 -1e-8，否則往上 +1e-8）
            tol = 1e-11
            eps = 1e-10
            mask_close = np.isclose(FC[:, 1], 1.0, atol=tol)
            FC[mask_close & (FC[:, 1] > 1.0), 1] -= eps
            FC[mask_close & ~(FC[:, 1] > 1.0), 1] += eps

            wi = FC[:,0]#提取所有頻率點
            ki=FC[:,1]#提取所有增益大小    #     gain > 1    ->  lower bounded case
                                                                                        # 0 < gain < 1    ->  upper bounded case
                                                                                        # 計算每個頻率點的幅度和相位
            #求出每個系統在頻率點的大小和相位
            magP, _ , _ = ctrl.bode(P, wi, plot=False)
            magM, phaseM, _ = ctrl.bode(M, wi, plot=False)
            magN, phaseN, _ = ctrl.bode(N, wi, plot=False)
            magX, phaseX, _ = ctrl.bode(X, wi, plot=False)
            magW, phaseW, _ = ctrl.bode(W, wi, plot=False)
            phaseM = np.rad2deg(phaseM)
            phaseN = np.rad2deg(phaseN)
            phaseX = np.rad2deg(phaseX)
            phaseW = np.rad2deg(phaseW)
            # 將幅度和相位轉換為複數形式
            valueM = magM * (np.cos(np.deg2rad(phaseM)) + 1j * np.sin(np.deg2rad(phaseM)))
            valueN = magN * (np.cos(np.deg2rad(phaseN)) + 1j * np.sin(np.deg2rad(phaseN)))
            valueX = magX * (np.cos(np.deg2rad(phaseX)) + 1j * np.sin(np.deg2rad(phaseX)))
            valueW = magW * (np.cos(np.deg2rad(phaseW)) + 1j * np.sin(np.deg2rad(phaseW)))
        
            for i in range(Nc) :
                weight[i]= 0.0000005
                rho = np.zeros((L, 1), dtype=complex)#創建空向量
                omevector = np.zeros((L, 1))
                for j in range(L):
                    rho[j] = np.exp(-j * wi[i] * 1j * Ts) 
                    omevector[j] = np.cos(j* wi[i] * Ts)
                Omega[:,:,i] = scipy.linalg.toeplitz(omevector)
                b = np.real(np.conj(valueW[i]) * valueN[i] * rho).ravel()
                c = np.real(np.conj(valueX[i]) * valueM[i] * rho).ravel()

                beta[:,i] = (-2 * (ki[i]**2 * b + magP[i]**2 * c)) / ((ki[i]**2 - 1) * magN[i]**2)
                gamma[i] = ((ki[i]**2 * np.abs(valueW[i])**2) - (magP[i]**2 * np.abs(valueX[i])**2)) / ((ki[i]**2 - 1) * magN[i]**2)


            allFrequencyconstraints = {}
            allFrequencyconstraints['alpha'] = alpha
            allFrequencyconstraints['beta'] = beta
            allFrequencyconstraints['Omega'] = Omega
            allFrequencyconstraints['Nc'] = Nc
            allFrequencyconstraints['gamma'] = gamma
            allFrequencyconstraints['weight'] = weight
            allFrequencyconstraints['w0'] = w0

            return allFrequencyconstraints

        Lq=self.parameter.Lq
        model=self.model
        model.remove(model.getQConstrs())
        Tcoe2_t, Tcoe1_t,Tcoe0_t=Time_domain_coefficient(path, path_index)
        allFC=Frequence_domain_coefficient(FC)
        q = model.addMVar(shape=Lq, vtype=GRB.CONTINUOUS, name="q")
        
        #加入目標函數
        obj =q @  Tcoe2_t @ q - 2*Tcoe1_t @ q + Tcoe0_t
        model.setObjective(obj, GRB.MINIMIZE)

        #加入限制條件
        for fi in range(allFC['Nc']):
            Omega_i = allFC['Omega'][:,:,fi]
            beta_i = allFC['beta'][:,fi]
            gamma_i = allFC['gamma'][fi]
            model.addConstr(q @ Omega_i @ q + beta_i @ q + gamma_i <= 0)
        # 優化模型
        model.optimize()
        #得到解
        if(model.status==GRB.OPTIMAL):
            status="Solved"
            solution = np.array(q.x).reshape((Lq, 1))
        else:
            status="Infeasible"
            solution=0
        return status, solution

    def LFTExpandedSS(self, Qnum):#組合q跟Jy成控制器
        ###邵平控制器
        #Qnum=[8885.431062041, -5053.406454447, 4563.933542546, 731.929927986, 586.1330973414]
        ###
        lq=self.parameter.Lq
        J=self.plant['Jy']
        qa1 = np.zeros(lq-1)
        qa2 = np.zeros(lq-1)
        qa1[1] = 1
        QA = scipy.linalg.toeplitz(qa1, qa2)
        QB = np.zeros((lq-1, 1))
        QB[0] = 1
        QC = np.array(Qnum[1:]).reshape(1, -1)
        QD = Qnum[0]
        F = 1 / (1 - QD * J.D[1, 1])
        Dk = F * QD
        Ck = J.C[0, :] + F * QD * J.C[1, :] 
        Ck=np.hstack([Ck, F * QC])
        Bk = J.B[:, 0] + J.B[:, 1] * F * QD 
        Bk = np.vstack([Bk,QB+ QB * J.D[1, 1] * F * QD])
        Ak11=J.A[:,:]+J.B[:,1]*F*QD*J.C[1,:]
        Ak12=J.B[:,1]*F*QC
        Ak21=QB*J.C[1,:]+QB*J.D[1,1]*F*QD*J.C[1,:]
        Ak22=QA+QB*J.D[1,1]*F*QC
        Ak=np.vstack([
                np.hstack([Ak11, Ak12]),
                np.hstack([Ak21, Ak22])
                ])
        CC = ctrl.ss(Ak, Bk, Ck, Dk,self.plant['Ts'])#控制器狀態空間
        return CC

    def switch_controller(self, path, path_index, FC, ek):
        # === 每個 Step 開始時，統一遞減共振紀錄表所有延遲步數 > 0 的項目 ===
        for col in range(self.resonanceTable.shape[1]):
            if self.resonanceTable[0, col] == 0:  # 空欄位，跳過
                break
            if self.resonanceTable[3, col] is not None and self.resonanceTable[3, col] > 0:
                self.resonanceTable[3, col] -= 1
        
        def suppress_resonance(original_status, Q, FC, path, path_index, ek, tolfreqResonances=10, reduceMag=10, increaseMag=5):
            def find_working_target(freq, start_target):
                for target in range(start_target - reduceMag, start_target, increaseMag):
                    test_FC = np.vstack([FC, [freq, 10 ** (target / 20)]])
                    test_FC = test_FC[np.argsort(test_FC[:, 0])]
                    status, _ = self.optimizationcvx(path, path_index, test_FC)
                    if status == "Solved":
                        return target, True
                return None, False

            manual_add_FC= np.empty((0, 2))

            # 沒有誤差資料或 PPO QCQP 無解時直接返回
            if np.sum(ek) == 0 or original_status != "Solved":
                return original_status, Q, manual_add_FC

            #找出潛在共振頻率
            path_district=path[path_index : path_index+self.pdl]
            CC=self.LFTExpandedSS(Q)
            OLoop=ctrl.minreal( ctrl.ss2tf(CC * self.plant["v2p"]), tol=1e-3, verbose=False)
            HFRw,_,_=find_resonance(CC, self.plant["v2p"], self.plant['Ts'], path_district, ek)
            HFRw = np.array(HFRw)#準備做向量運算
            
            #拿潛在共振對應紀錄表上以記錄的共振
            saved_freqs = self.resonanceTable[0, :]#取出已儲存的共振頻率
            diff_matrix = np.abs(HFRw[:, np.newaxis] - saved_freqs)  #廣播運算shape = (len(HFRw), len(res_freqs))
            match_indices = np.where(diff_matrix < tolfreqResonances, True, False) #是一個 2D 布林矩陣，表示「HFRw 中的每個頻率」與「resonanceTable 中的每個頻率」是否匹配（差距 < tolfreqResonances）。
            has_match = np.any(match_indices, axis=1)# 布林索引，每一列有沒有至少一個True，代表共振頻率已登記

            # 新增沒登記過的共振（直接加到正式共振表）
            unsaved_freqs = HFRw[~has_match]
            for freq in unsaved_freqs:
                insert_pos = np.searchsorted(self.resonanceTable[0, :], freq)
                new_col = np.array([
                    [freq],#頻率
                    [2],#檢查真共振步數
                    [None],#Target
                    [1],#延遲步數
                    [True],#標記為可加入
                ])
                if self.resonanceTable[0,0] == 0: self.resonanceTable = new_col
                else: self.resonanceTable = np.hstack((self.resonanceTable[:, :insert_pos], new_col, self.resonanceTable[:, insert_pos:]))
            
            # 建立暫存共振表，後續 target 設定先寫到暫存表
            tempTable = self.resonanceTable.copy()

            # 處理這輪偵測到的所有共振（重新計算 match_indices 以取得正確的欄位索引）
            saved_freqs = tempTable[0, :]
            diff_matrix = np.abs(HFRw[:, np.newaxis] - saved_freqs)
            match_indices = np.where(diff_matrix < tolfreqResonances, True, False)
            for i in range(len(HFRw)):
                idx = np.where(match_indices[i, :])[0][0]  # HFRw[i] 對應 tempTable 的哪一欄
                if tempTable[1, idx] == 0:#檢查真共振步數為0，確認為真共振，開始設定target
                    if tempTable[2, idx]==None:#第一次壓制未決定target（局部可行性測試）
                        complex_mag, _, _ = ctrl.freqresp(OLoop, HFRw[i])
                        loopgain = 0 if 20 * np.log10(np.abs(complex_mag)) > 0 else int(20 * np.log10(np.abs(complex_mag)))
                        target, success = find_working_target(HFRw[i], loopgain)
                        if success:
                            tempTable[2, idx] = target#設定Target
                        else:
                            tempTable[2, idx] = loopgain+1#設定Target
                            tempTable[3, idx]=0#延遲步數設0
                    elif tempTable[3, idx]==0:#延遲步數為0，還是有共振（局部可行性測試）
                        old_target=tempTable[2, idx]
                        target, success = find_working_target(HFRw[i], old_target)
                        if success:
                            tempTable[2, idx] = target#設定Target
                            tempTable[3, idx]=1#延遲步數設1
                else: tempTable[1, idx] -= 1#檢查真共振步數減1

            # 根據暫存表以記錄的目標增益設定手動加入的 FC  
            for i in range(tempTable.shape[1]):
                if tempTable[0,i]==0:break
                target = tempTable[2, i]
                if target is not None:
                    freq = tempTable[0, i]
                    mag = 10 ** (target / 20)
                    manual_add_FC = np.vstack([manual_add_FC, [freq, mag]])

            #神經網路跟手動FC合併       
            combined= np.vstack([FC, manual_add_FC])
            new_FC = combined[np.argsort(combined[:, 0])]#排序

            # 最終 RSS QCQP：把所有限制條件一起求解
            status, newQ=self.optimizationcvx(path, path_index, new_FC)

            if status=="Solved":
                # 最終 QCQP 有解，才把暫存表寫回正式共振紀錄表
                self.resonanceTable = tempTable
                return status, newQ, manual_add_FC
            else :
                # 最終 QCQP 無解，丟棄暫存表，共振紀錄表保持不變
                status="semiSolved"
                return status, Q, manual_add_FC

        status, Q=self.optimizationcvx(path, path_index, FC)
        path_district=path[path_index : path_index+self.pdl]
        status, Q, manual_add_FC=suppress_resonance(status ,Q, FC, path, path_index, ek)

        if  status=="Solved" :#成功解出Q
            self.CC=self.LFTExpandedSS(Q)
            self.last_solved_FC=FC.copy()
            self.X0_hat, ek_hat, _=SimulateResponse(path_district, self.CC, self.plant['v2p'], self.X0_hat, self.plant['Ts'])
        else:
            self.X0_hat, ek_hat, _=SimulateResponse(path_district, self.CC, self.plant['v2p'], self.X0_hat, self.plant['Ts'])
        return status, self.CC, ek_hat, manual_add_FC

    def reward(self, FC, status, CC, ek,  visual=0):#計算成本函數

        def FCfreq(FC, Wgc):
            #用對數間距計算 FC 分布均勻度，(max_gap - min_gap) * scale_weigh
            Nc1 = FC.shape[0]
            Nc2 = np.sum(FC[:, 0] < Wgc)  # 小於Wgc的個數
            
            # 收集 Wgc 以後的所有對數間距
            log_gaps = []
            for i in range(Nc1-1, Nc2-1, -1):  # Wgc以後的分布
                if FC[i-1, 0] <= 0 or FC[i, 0] <= 0:
                    continue
                log_gap = np.log10(FC[i, 0]) - np.log10(FC[i - 1, 0])
                log_gaps.append(log_gap)
            
            if len(log_gaps) < 2:
                return 0
            
            max_gap = max(log_gaps)
            min_gap = min(log_gaps)
            return (max_gap - min_gap)

        Wgc=find_Wgc(CC, self.plant["G"],self.plant["Ts"])
        parameter=self.parameter
        cost=0
        
        if  status=="Solved" or status=="semiSolved":#成功解出Qnum
            #Error
            weights = 0.7 ** np.arange(len(ek))  # 產生一個 0.7**i 的數列
            sumError=np.sum(abs(ek) * weights)
            cost_sumError = parameter.w_sumError * sumError
            cost = cost  + cost_sumError

            #FCfreq, constraints distance
            FCfreq=FCfreq(FC, Wgc)
            cost_FCfreq=parameter.w_FCfreq * FCfreq
            cost = cost  + cost_FCfreq

            cost_Wgc=0
            if status=="semiSolved":
                cost_Wgc=parameter.w_Wgc * Wgc
                cost = cost  + cost_Wgc

            reward=1e+5/cost
        else:#沒有解出Qnum，使用舊的CLoop
            #region
            refference_FC=20*np.log10(self.last_solved_FC)
            FC=20*np.log10(FC)#轉成神經網路的輸出
            diff=abs(FC-refference_FC)#計算神經網路輸出和參考頻率點相差
            reward = -np.sum(diff*self.parameter.w_earlyTrain)
            sumError=FCfreq=cost_sumError=cost_FCfreq=cost_Wgc=0
            #endregion

        if(visual>=1):
            index = ["sumError","FCfreq","Wgc","status"]
            data={
                "Vlaue":[sumError, FCfreq, Wgc, status],
                "Cost":[cost_sumError, cost_FCfreq, cost_Wgc, reward]
             }
            data = pd.DataFrame(data,index=index)
            print(f"{data} \n")
        return reward

class PlotExporter:
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
        plt.figure(figsize=(12, 6.08))
        OLoop = ctrl.minreal(ctrl.ss2tf(CC * plant), tol=1e-3, verbose=False)
        mag, _, oma = ctrl.bode(OLoop, dB=True, omega_limits=[1e-2, 3e3], plot=False)
        #畫OLoop
        plt.plot(oma, 20 * np.log10(mag), color='b')
        #畫手動添加FC點
        plt.scatter(manual_add_FC[:, 0], 20 * np.log10(manual_add_FC[:, 1]), color='g')
        #畫Actor產生FC點
        plt.scatter(FC[:, 0], 20 * np.log10(FC[:, 1]), color='r')
        #畫全部FC連線
        combined= np.vstack([FC, manual_add_FC])#神經網路跟手動FC合併
        new_FC = combined[np.argsort(combined[:, 0])]#排序
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

    def plot_error(self, error):
        combined_data = np.concatenate(error)
        if  not np.any(np.abs(combined_data) > 30):
            plt.ylim(-30, 30)
        print("RMS Error: ",np.sqrt(np.mean(combined_data**2)))
        x = np.linspace(0, len(combined_data)*0.001, len(combined_data))
        plt.plot(x,combined_data)
        plt.title("Error", fontsize=18)
        plt.xlabel("time(s)", fontsize=14)
        plt.ylabel("Magnitude(um)", fontsize=14)
        plt.grid()
        plt.tight_layout()
        plt.show()

    def save_mp4(self):
        """將所有儲存的圖片製作成 MP4"""
        with imageio.get_writer(self.video_name, fps=self.fps, codec='libx264', quality=8) as writer:
            for filename in self.saved_frames:
                image = imageio.imread(filename)
                writer.append_data(image)
        self.saved_frames=[]
        self.step=0

        print(f"MP4 saved to {self.video_name}")
