import argparse
import os
import time
import warnings
from datetime import datetime

import control as ctrl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import socket
import torch

import CNC
import pc_server
from PPO_brain import PPO

np.set_printoptions(precision=10,suppress=True)#設置打印位數，科學記號
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
time_start=time.time()

######   存取資料   ######
#region
read=True
use_switch_model = False  # True: 雙模型切換, False: 單模型
if use_switch_model:
    read_file_name1='Modelunc1.pth'  # 追蹤模型
    read_file_name2='Modelran1.pth'  # 共振模型
else:
    read_file_name='Modelunc.pth'
#endregion

######   參數區域    ######
 #region
#PPO參數
n_states=131# action 28, path FFT 100, maxResonance 2, error 1
numFC=14#頻率點數量
bound= 20*np.log10(3000)#神經網路輸出限制dB
parser1 = argparse.ArgumentParser(description="PPO參數")
parser1.add_argument('--iteration', type=int, default=2000)
parser1.add_argument('--n_step_learning', type=int, default=20)
parser1.add_argument('--learning_rate', type=float, default=1e-5)
parser1.add_argument('--mini_batch', type=int, default=30)
parser1.add_argument('--batch_size', type=int, default=2000)
parser1.add_argument('--n_round_batch', type=int, default=60)
parser1.add_argument('--gamma', type=float, default=0.9)
parser1.add_argument('--epsilon', type=float, default=0.03)
parser1.add_argument('--c_update_steps', type=int, default=10)
parser1.add_argument('--a_update_steps', type=int, default=3)
PPO_parameter = parser1.parse_args()
#CNC參數
x_polegain=0.4352
z_polegain=0.4952
parser2 = argparse.ArgumentParser(description="CNC參數")
parser2.add_argument('--Lq', type=int, default=10)
parser2.add_argument('--w_sumError', type=float, default=1e+2)
parser2.add_argument('--w_FCfreq', type=float, default=1e+0)
parser2.add_argument('--w_Wgc', type=float, default=1e+3)
parser2.add_argument('--w_earlyTrain', type=float, default=5e-3)
CNC_parameter = parser2.parse_args()
#其他參數
Ts=0.001
pdl=300#path_distric_len 多少ms一個區間
fft_limit_freq = 2  # path_FFT 取到幾 Hz
num_low_freq_FC = 3  # 低頻限制點數量
manual_FC = np.array([
    [0.1, 1000],
    [1, 100],
    [10, 10],
    [100, 1.1],
    [300, 0.8],
    [500, 0.39],
    [700, 0.16],
    [900, 0.07],
    [1000, 0.08],
    [1300, 0.1],        
    [1500, 0.09],
    [2000, 0.03],
    [2500, 0.07],
    [3000, 0.1]
])
#TCP參數
HOST = "0.0.0.0"
PORT = 5005
#endregion

######     函數區     ######
#region
def Time_show(time1, time2):
    total_time=int(time2-time1)
    second=total_time%60
    total_time=int(total_time/60)
    minent=total_time%60
    hour=int(total_time/60)
    print(hour,":",minent,":",second)
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
    magnitude = np.abs(yf[:N // 2])/pdl#頻譜振幅
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
def state_action(FC):
    return 20*np.log10(np.hstack((FC[:, 0], FC[:, 1])).ravel())/30
def state_maxResonance(CC, plant, Ts, path_district, ek):
    HFRw, HFRm, HFRg=CNC.find_resonance(CC, plant, Ts, path_district, ek)
    if len(HFRw) > 0:
        idx = np.argmax(HFRm)
        max_resonance_freq_dB = 20 * np.log10(HFRw[idx])/30
        max_resonance_gain_dB = HFRg[idx]/30
    else:
        max_resonance_freq_dB = 0
        max_resonance_gain_dB = 0
    return [max_resonance_freq_dB, max_resonance_gain_dB]
def state_error(ek):
    weights = 0.7 ** np.arange(len(ek))  # 產生一個 0.7**i 的數列
    sumError=np.sum(abs(ek) * weights)
    return   [np.log1p(sumError)/10]
def reset_buffers(num_path_district, pdl):
    data_buffer = [None] * num_path_district  # 儲存計算reward前的資料
    ek_buffer = np.zeros((3, pdl))            # 模擬誤差延遲進入
    X0 = 0                                     # 初始狀態
    iter_r = 0                                 # 累計 reward
    error = []                                 # 儲存每步誤差
    return data_buffer, ek_buffer, X0, iter_r, error
#endregion

######   創造實例    ######
#region
if use_switch_model:
    NO1agent = PPO(n_states , numFC*2, bound, PPO_parameter, device)  # 追蹤模型
    NO2agent = PPO(n_states , numFC*2, bound, PPO_parameter, device)  # 共振模型
else:
    agent = PPO(n_states , numFC*2, bound, PPO_parameter, device)

model_x = CNC.CNCModel('x',Ts)#創建馬達實例
path_model=CNC.PathModel(Ts)
ID_Plant=model_x.ID_Plant()#取得馬達ID模型
testpath=path_model.test_path()
testpath2=path_model.test_path2()
up_down_chirp=path_model.up_down_chirp()

costfunction_x=CNC.Costfunction(CNC_parameter, x_polegain, ID_Plant, testpath, pdl, numFC, num_low_freq_FC)#manual_FC.copy()
PlotExporter=CNC.PlotExporter()

srv=pc_server.create_server(HOST, PORT)
#endregion

#####   讀取資料   ####
#region
if read:
    try:
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        
        if use_switch_model:
            # 雙模型模式
            checkpoint1 = torch.load(f"../Model/{read_file_name1}",map_location="cpu",weights_only=False)
            checkpoint2 = torch.load(f"../Model/{read_file_name2}",map_location="cpu",weights_only=False)
            max_iteration1 = max(int(key.split(':')[1]) for key in checkpoint1.keys())
            max_iteration2 = max(int(key.split(':')[1]) for key in checkpoint2.keys())
            user_input = input(f"追跡Model最多為{max_iteration1}輪。共振Model最多為{max_iteration2}輪。請輸入m,n你想要從哪一輪開始或默認最大輪數: ")
            if user_input.strip() == "":
                start_iteration1 = max_iteration1
                start_iteration2 = max_iteration2
            else:
                try:
                    parts = user_input.split(",")
                    start_iteration1 = int(parts[0].strip())
                    start_iteration2 = int(parts[1].strip())
                except (ValueError, IndexError):
                    print("輸入格式錯誤，改用預設值。")
                    start_iteration1 = max_iteration1
                    start_iteration2 = max_iteration2

            iteration_key1 = f'iteration:{start_iteration1}'
            iteration_key2 = f'iteration:{start_iteration2}'
            NO1agent.actor_model.load_state_dict(checkpoint1[iteration_key1]['actor'])
            NO2agent.actor_model.load_state_dict(checkpoint2[iteration_key2]['actor'])
            costfunction_x.set_controller(checkpoint1[iteration_key1]['FC'])
            print(f"載入model成功: 追蹤模型第{start_iteration1}輪, 共振模型第{start_iteration2}輪")
        else:
            # 單模型模式
            checkpoint = torch.load(f"../Model/{read_file_name}",map_location="cpu",weights_only=False)
            if checkpoint and all('iteration:' in key for key in checkpoint.keys()):
                max_iteration = max(int(key.split(':')[1]) for key in checkpoint.keys())
                user_input = input(f"現在為{max_iteration}輪。請輸入你想要從哪一輪開始 (默認是 {max_iteration}): ")
                if user_input.strip() == "":
                    start_iteration = max_iteration
                else:
                    try:
                        start_iteration = int(user_input)
                        if start_iteration > max_iteration:
                            print(f"輸入的數字超過了最大迭代次數，將從最大輪次 {max_iteration} 開始。")
                            start_iteration = max_iteration
                    except ValueError:
                        print("輸入無效，將從最大輪次開始。")
                        start_iteration = max_iteration
                iteration_key = f'iteration:{start_iteration}'
                agent.actor_model.load_state_dict(checkpoint[iteration_key]['actor'])
                agent.critic_model.load_state_dict(checkpoint[iteration_key]['critic'])
                status, _,_,_=costfunction_x.switch_controller(testpath,0, checkpoint[iteration_key]['FC'], [0])
                costfunction_x.set_controller(checkpoint[iteration_key]['FC'])
                print(f"載入model {read_file_name} 第{iteration_key}輪，初始限制條件{status}")
    except FileNotFoundError:
        if use_switch_model:
            print(f"找不到model檔案")
            os._exit(1)
        else:
            start_iteration = 0
            print(f"找不到model檔案")
else:
    if use_switch_model:
        print("雙模型模式必須讀取model")
        os._exit(1)
    else:
        start_iteration = 0
        print(f"不引進model，從0訓練")
#endregion

#創造空容器
#region
FC = np.zeros((numFC, 2))
# 數據收集容器（只存每步的關鍵數據）
data_collector = {
    'CC_list': [],
    'FC_list': [],
    'manual_FC_list': [],
    'error_list': [],
    'status_list': [],
    'resonance_freq_list': [],
    'resonance_gain_list': []
}
data_collector['model_used'] = []  # 記錄每步使用的model (單模型時永遠是 model1)
#endregion

#邵平控制器
shaoping = [8885.431062041,       -24637.015400412543,     31725.764043065406,
 -24272.73172976235,      10722.74306304059  ,    -2552.3219619078864,
    745.8723434910148,     -332.2659082659679,        1.,
     -1.344771898714865,      0.551909622794037  ,   -0.040436304715828,
     -0.093678503305545,     -0.018232768906639  ,   -0.011387297705785,
     -0.]
CC_shaoping = ctrl.tf2ss(ctrl.TransferFunction(shaoping[:8], shaoping[8:], Ts))

#中央控制器
X_central = [1861.7156, -2631.6055, 1055.3655, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             1., -1.2839379, 0.6899362, -0.10736613, 0., 0., 0., 0., 0., 0., 0., 0., 0.]
CC_X_central = ctrl.tf2ss(ctrl.TransferFunction(X_central[:13], X_central[13:], Ts))

#X軸本身機台共振測試控制器(舊)
X_resonance_old=[1.24776671e+04, -2.82535941e+04, 2.20793412e+04, -6.01791813e+03,
           -2.36084310e-02, 2.39126305e-01, -5.14901653e-01, 3.92582295e+03,
           -9.47329155e+03, 7.77157149e+03, -2.22327578e+03, -5.28287522e-01,
           -1.99142081e-02, 1.00000000e+00, -1.36936611e+00, 4.47522069e-01,
           -1.07366429e-01, -3.20512604e-07, -6.61145311e-07, -2.30769282e-06,
           -5.56249046e-06, -2.68773333e-02, -7.62712047e-02, -2.07301160e-05,
           -6.82492589e-07, 6.03972590e-23]
CC_X_resonance_old = ctrl.tf2ss(ctrl.TransferFunction(X_resonance_old[:13], X_resonance_old  [13:], Ts))

#X軸本身機台共振測試控制器(新) omega=800, zeta=0.05
# gain=3 (小共振)
X_resonance_g3 = [1861.715550729014, -5062.418655889311, 6145.8167941565935, -3716.5382114685676, 937.8455547143777,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  1.0, -2.6240934309635646, 3.3337290483473847, -2.217212039534731, 0.7807787244052766,
                  -0.0991114316607929, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
CC_X_resonance_g3 = ctrl.tf2ss(ctrl.TransferFunction(X_resonance_g3[:15], X_resonance_g3[15:], Ts))

# gain=6 (中小共振)
X_resonance_g6 = [1861.7155507290172, -4869.892976645662, 5681.148773381589, -3335.257315036361, 828.7069998082437,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  1.0, -2.624093430963564, 3.333729048347383, -2.2172120395347292, 0.7807787244052758,
                  -0.09911143166079275, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
CC_X_resonance_g6 = ctrl.tf2ss(ctrl.TransferFunction(X_resonance_g6[:15], X_resonance_g6[15:], Ts))

# gain=9 (中等共振)
X_resonance_g9 = [1861.7155507290104, -4677.367297445485, 5216.480752690237, -2953.9764186765537, 719.5684449287203,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  1.0, -2.6240934309635646, 3.3337290483473843, -2.2172120395347314, 0.7807787244052766,
                  -0.09911143166079292, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
CC_X_resonance_g9 = ctrl.tf2ss(ctrl.TransferFunction(X_resonance_g9[:15], X_resonance_g9[15:], Ts))

# gain=12 (大共振)
X_resonance_g12 = [1861.715550729013, -4484.841618155055, 4751.812731883474, -2572.695522236054, 610.4298900356185,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   1.0, -2.6240934309635646, 3.3337290483473843, -2.2172120395347314, 0.7807787244052766,
                   -0.09911143166079292, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
CC_X_resonance_g12 = ctrl.tf2ss(ctrl.TransferFunction(X_resonance_g12[:15], X_resonance_g12[15:], Ts))

#實驗設定
path=testpath
path_index=0
num_path_district=int((len(path)-path_index)/pdl)
prev_dominant_freq=0.1
if use_switch_model:
    switch=False
    resonance_detected=0
#暫存區歸零
costfunction_x.initialize()
data_buffer, ek_buffer, X0, iter_r, error = reset_buffers(num_path_district, pdl)
ek=np.zeros(pdl)
CC=None
#準備第一個state
path_FFT_magnitude, prev_dominant_freq, _=path_FFT(path, path_index , prev_dominant_freq)
last_solved_FC=costfunction_x.last_solved_FC
s=np.concatenate([state_action(last_solved_FC), path_FFT_magnitude, np.zeros(3)])#合成第一組狀態
print("Standby for link in \n")

for step in range(num_path_district+1):#開始產生歷程，因為Error有延遲所以要多兩步收集資料
    try:
        #等待連線（10秒超時）
        srv.settimeout(10.0)
        conn, addr = srv.accept()
        #從labview收ek
        ek = pc_server.recv(conn)
        if ek is None or len(ek) != pdl:
            print("[ERROR] recv failed or wrong length, saving data and exit...")
            break
    except socket.timeout:
        print(f"[WARNING] Accept timeout at step {step}, no LabVIEW connection for 10s, saving data and exit...")
        break
    except Exception as e:
        print(f"[ERROR] Connection error at step {step}: {e}, saving data and exit...")
        break

    #建構state並收集error
    if step>=1:
        data_collector['error_list'].append(ek.copy())
        if step==num_path_district:
            break
        path_FFT_magnitude, prev_dominant_freq, _=path_FFT(path, path_index, prev_dominant_freq)
        s = np.concatenate([ state_action(FC), path_FFT_magnitude, state_maxResonance(CC, ID_Plant["v2p"], Ts, path_district, ek), state_error(ek)])

        #雙模型模式：檢測共振換model
        if use_switch_model:
            ##
            print(state_maxResonance(CC, ID_Plant["v2p"], Ts, path_district, ek_buffer[step%3])[0])
            ##
            if state_maxResonance(CC, ID_Plant["v2p"], Ts, path_district, ek)[0]!=0:
                resonance_detected+=1

    #產生動作
    if use_switch_model:
        if resonance_detected>=1: 
            if switch==False:
                print("Switch Model : step ", step,"\n") 
                switch=True
            a = np.array(NO2agent.choose_action(s))  # 共振模型
            data_collector['model_used'].append('NO2_resonance')
        else: 
            a = np.array(NO1agent.choose_action(s))  # 追蹤模型
            data_collector['model_used'].append('NO1_tracking')
    else:
        a = np.array(agent.choose_action(s))  # 單模型
        data_collector['model_used'].append('model1')
    
    action = 10.0 ** (a / 20.0)               # 線性倍率
    FC[:, 0] = action[:numFC]                 # 頻率
    FC[:, 1] = action[numFC:]                 # 增益
    sorted_indices = np.argsort(FC[:, 0])  #排序FC的索引
    FC = FC[sorted_indices]  # 按照索引排序FC
    #是否合成新控制器並模擬運行
    status, CC, ek_hat, manual_add_FC=costfunction_x.switch_controller(path, path_index, FC.copy(), ek)
    path_district=path[path_index : path_index+pdl]
    
    #使用固定控制器
    CC=CC_X_resonance

    #收集實驗數據（每步）
    data_collector['CC_list'].append(CC.copy())
    data_collector['FC_list'].append(FC.copy())
    data_collector['manual_FC_list'].append(manual_add_FC.copy())
    data_collector['status_list'].append(status)

    #轉出CC
    CC_tf = ctrl.ss2tf(CC)
    # 注意：python-control 回傳的是 [a0, a1, ..., aN] / [b0, b1, ..., bN]
    den = np.array(CC_tf.den[0][0])  # a0, a1, ..., aN
    num = np.array(CC_tf.num[0][0])  # b0, b1, ..., bN
    # 如果 LabVIEW 端要求 num、den 長度一樣，可以這樣補 0
    if len(num) < len(den):
        num = np.pad(num, (0, len(den) - len(num)))
    CCdata=np.concatenate((num, den))
    #TCP傳輸資料
    try:
        ok = pc_server.send(conn, CCdata)
        if not ok:
            print("[ERROR] send failed, saving data and exit...")
            break
    except Exception as e:
        print(f"[ERROR] Send error at step {step}: {e}, saving data and exit...")
        break

    #準備下一步
    path_index=path_index+pdl

    #儲存共振資料並打印
    resonance_state = state_maxResonance(CC, ID_Plant["v2p"], Ts, path_district, ek)
    freq_normalized = resonance_state[0]
    mag_normalized = resonance_state[1]
    # 转换为线性频率和dB值存储
    freq_linear = 10 ** ((freq_normalized / 20)*30) if freq_normalized != 0 else 0
    mag_dB = mag_normalized * 30
    data_collector['resonance_freq_list'].append(freq_linear)
    data_collector['resonance_gain_list'].append(mag_dB)
    print(
        f"{step:4d} | "
        f"{status:<11} | "
        f"{freq_linear:10.5g} | "
        f"{mag_dB:12.5g}"
        )
 
#關閉server
try:
    conn.close()
except:
    pass

#計算實驗時長
time_end = time.time()
experiment_duration = time_end - time_start

#整理完整實驗數據
experiment_datetime = datetime.fromtimestamp(time_start).strftime("%Y.%m.%d.%H.%M")
experiment_data = {
    # 1. 實驗基本資訊
    'execution_script': 'Runtime.py',
    'experiment_datetime': datetime.fromtimestamp(time_start).strftime("%Y-%m-%d %H:%M:%S"),
    'experiment_duration': experiment_duration,
    'actual_steps': len(data_collector['CC_list']),
    'use_switch_model': use_switch_model,
    
    # 2. 系統參數
    'Ts': Ts,
    'pdl': pdl,
    'numFC': numFC,
    'fft_limit_freq': fft_limit_freq,
    'num_low_freq_FC': num_low_freq_FC,
    'x_polegain': x_polegain,
    'z_polegain': z_polegain,
    'bound': bound,
    'n_states': n_states,
    'CNC_params': {
        'Lq': CNC_parameter.Lq
    },
    
    # 3. 參考路徑資訊
    'reference_path': path,
    'path_length': len(path),
    'num_districts': num_path_district,
    
    # 4. 受控體模型
    'ID_Plant_v2p': ID_Plant["v2p"]
}

# 5. Model資訊區：統一結構
if use_switch_model:
    experiment_data['model1_filename'] = read_file_name1
    experiment_data['model2_filename'] = read_file_name2
    experiment_data['model1_iteration'] = start_iteration1
    experiment_data['model2_iteration'] = start_iteration2
else:
    experiment_data['model1_filename'] = read_file_name
    experiment_data['model2_filename'] = None
    experiment_data['model1_iteration'] = start_iteration
    experiment_data['model2_iteration'] = None

# 6. 每步數據
experiment_data['CC_list'] = np.array(data_collector['CC_list'], dtype=object)
experiment_data['FC_list'] = np.array(data_collector['FC_list'])
experiment_data['manual_FC_list'] = np.array(data_collector['manual_FC_list'], dtype=object)
experiment_data['error_list'] = np.array(data_collector['error_list'])
experiment_data['status_list'] = np.array(data_collector['status_list'])
experiment_data['resonance_freq_list'] = np.array(data_collector['resonance_freq_list'])
experiment_data['resonance_gain_list'] = np.array(data_collector['resonance_gain_list'])
experiment_data['model_used'] = data_collector['model_used']
experiment_data['switch_step'] = None if not use_switch_model else (None if not switch else next((i for i, m in enumerate(data_collector['model_used']) if m == 'NO2_resonance'), None))

#創建保存目錄
save_dir = os.path.join("..", "ExperimentData", experiment_datetime)
os.makedirs(save_dir, exist_ok=True)

#保存數據
save_path = os.path.join(save_dir, "runtime_data.npz")
np.savez_compressed(save_path, **experiment_data, allow_pickle=True)
print(f"\n實驗數據已保存至: {save_path}")
print(f"實驗時長: {int(experiment_duration//60)}分{int(experiment_duration%60)}秒")

#做GIF（使用data_collector中的數據）
for i in range(len(data_collector['CC_list'])):
    PlotExporter.plot_frame(
        data_collector['CC_list'][i],
        ID_Plant["v2p"],
        data_collector['FC_list'][i],
        data_collector['manual_FC_list'][i]
    )
    PlotExporter.save_mp4()
#畫出誤差
PlotExporter.plot_error(data_collector['error_list'])
