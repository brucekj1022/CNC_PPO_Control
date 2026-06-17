import argparse
import os
import time
import warnings
from datetime import datetime

import control as ctrl
import numpy as np
import scipy
import torch

import CNC
from PPO_brain import PPO, ReplayBuffer

np.set_printoptions(precision=5,suppress=True)#設置打印位數，科學記號
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
time_start=time.time()

######   存取資料   ######
#region
read=True
use_switch_model = False # True: 雙模型切換, False: 單模型
if use_switch_model:
    read_file_name1='ModelBUE1.pth'  # 追蹤模型
    read_file_name2='ModelPRE1.pth'  # 共振模型
else:
    read_file_name='ModelPRE1.pth'
#endregion

######   參數區域    ######
 #region
#PPO參數
n_states=131# action 28, path FFT 100, maxResonance 2, error 1
numFC=14#頻率點數量
bound= 20*np.log10(3000)#神經網路輸出限制dB
parser1 = argparse.ArgumentParser(description="PPO參數")
parser1.add_argument('--iteration', type=int, default=3000)
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
parser2.add_argument('--w_sumError', type=float, default=1e+3)
parser2.add_argument('--w_FCfreq', type=float, default=1e+0)
parser2.add_argument('--w_Wgc', type=float, default=1e+3)
parser2.add_argument('--w_earlyTrain', type=float, default=5e-3)
CNC_parameter = parser2.parse_args()
#其他參數
Ts=0.001
pdl=300#path_distric_len 多少ms一個區間
fft_limit_freq = 15  # path_FFT 取到幾 Hz
num_low_freq_FC = 3  # 低頻限制點數量
max_error_um = 10000  # 最大誤差 (um)，超過視為不穩定
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
        max_resonance_freq_dB = 20 * np.log10(HFRw[idx])/30  # 归一化给神经网络
        max_resonance_gain_dB = HFRg[idx]/30  # 归一化给神经网络
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
training_path=path_model.training_path()
testpath=path_model.test_path()
testpath2=path_model.test_path2()
up_down_chirp=path_model.up_down_chirp()

costfunction_x=CNC.Costfunction(CNC_parameter, x_polegain, ID_Plant, testpath, pdl, numFC, num_low_freq_FC)#manual_FC.copy()
PlotExporter=CNC.PlotExporter()
#endregion

#####   讀取資料   ####
 #region
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
                start_iteration1= max_iteration1
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
            status, _,_,_=costfunction_x.switch_controller(testpath,0, checkpoint[iteration_key]['FC'], [0])
            costfunction_x.set_controller(checkpoint[iteration_key]['FC'])
            print(f"載入model {read_file_name} 第{iteration_key}輪，初始限制條件{status}")
except FileNotFoundError:
    print(f"找不到model檔案")
    os._exit(1)
 #endregion

#創造空容器
#region
FC = np.zeros((numFC, 2))
#endregion

#邵平控制器
CCnub = [8885.431062041,       -24637.015400412543,     31725.764043065406,
 -24272.73172976235,      10722.74306304059  ,    -2552.3219619078864,
    745.8723434910148,     -332.2659082659679,        1.,
     -1.344771898714865,      0.551909622794037  ,   -0.040436304715828,
     -0.093678503305545,     -0.018232768906639  ,   -0.011387297705785,
     -0.]
CC_shaoping = ctrl.tf2ss(ctrl.TransferFunction(CCnub[:8], CCnub[8:], Ts))

#實驗設定
Plant=model_x.test_Plant()
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

#創建實驗資料收集器
experiment_datetime = datetime.now().strftime("%Y.%m.%d.%H.%M")
data_collector = {
    'CC_list': [],
    'FC_list': [],
    'manual_FC_list': [],
    'error_list': [],
    'status_list': [],
    'resonance_freq_list': [],
    'resonance_gain_list': [],
    'model_used': []  # 記錄使用的模型 (單模型時永遠是 model1)
}

#準備第一個state
path_FFT_magnitude, prev_dominant_freq, _=path_FFT(path, path_index , prev_dominant_freq)
last_solved_FC=costfunction_x.last_solved_FC
s=np.concatenate([state_action(last_solved_FC), path_FFT_magnitude, np.zeros(3)])#合成第一組狀態

for step in range(num_path_district+1):#開始產生歷程，因為Error有延遲所以要多兩步收集資料
    #判斷是否發散
    if np.sqrt(np.mean((ek_buffer[step%3])**2)) > max_error_um:#判斷Error有沒有發散
        num_path_district=step-1
        break
    if(step>=num_path_district):
        continue
    
    #產生動作
    if use_switch_model:
        if resonance_detected>=1 : 
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
    
    action = 10**(a/20) #從db轉自然數
    FC[:,0]=action[:numFC]#action前一半做為頻率 
    FC[:,1]=action[numFC:]#action後一半做為大小
    sorted_indices = np.argsort(FC[:, 0])  #排序FC的索引
    FC = FC[sorted_indices]  # 按照索引排序FC
    
    #是否合成新控制器並模擬運行
    status, CC, ek_hat, manual_add_FC=costfunction_x.switch_controller(path, path_index, FC.copy(), ek_buffer[step%3])
    path_district=path[path_index : path_index+pdl]

    #使用紹平控制器
    #CC=CC_shaoping

    X0, ek_buffer[(step+2)%3 , :], _=CNC.SimulateResponse(path_district.copy(), CC.copy(), Plant['v2p'], X0,Ts)#模擬輸出Controller給工具機
    PlotExporter.plot_frame(CC, ID_Plant['v2p'], FC, manual_add_FC)#製作Gif

    #資料存進data_buffer
    path_FFT_magnitude, prev_dominant_freq, _=path_FFT(path, path_index+pdl, prev_dominant_freq)
    s_ = np.concatenate([ state_action(FC), path_FFT_magnitude, state_maxResonance(CC, ID_Plant["v2p"], Ts, path_district, ek_buffer[(step+1)%3]), state_error(ek_buffer[(step+1)%3])])
    data_buffer[step]=(s.copy(), a.copy(), s_.copy(), FC.copy(), status, CC.copy(), path_district.copy(), ek_buffer[(step+2)%3 , :].copy())
    
    #收集實驗資料
    data_collector['CC_list'].append(CC.copy())
    data_collector['FC_list'].append(FC.copy())
    data_collector['manual_FC_list'].append(manual_add_FC.copy())
    data_collector['error_list'].append(ek_buffer[(step+2)%3].copy())
    data_collector['status_list'].append(status)
    resonance_state = state_maxResonance(CC, ID_Plant["v2p"], Ts, path_district, ek_buffer[step%3])
    # 转换为线性频率和dB值存储
    freq_linear = 10 ** ((resonance_state[0] / 20) * 30) if resonance_state[0] != 0 else 0
    mag_dB = resonance_state[1] * 30
    data_collector['resonance_freq_list'].append(freq_linear)
    data_collector['resonance_gain_list'].append(mag_dB)
    
    #準備下一步
    s=s_
    path_index=path_index+pdl
    if use_switch_model:
        if state_maxResonance(CC, ID_Plant["v2p"], Ts, path_district, ek_buffer[step%3])[0]!=0:
            resonance_detected+=1

    freq_normalized, mag_normalized = state_maxResonance(CC, ID_Plant["v2p"], Ts, path_district, ek_buffer[step%3])
    freq_linear = 10 ** ((freq_normalized / 20)*30) if freq_normalized != 0 else 0
    mag_dB = mag_normalized * 30
    print(
        f"{step:4d} | "
        f"{status:<11} | "
        f"{freq_linear:10.5g} | "
        f"{mag_dB:12.5g}"
        )


#計算實驗時長
time_end = time.time()
experiment_duration = time_end - time_start

#整理完整實驗數據
experiment_datetime = datetime.fromtimestamp(time_start).strftime("%Y.%m.%d.%H.%M")
experiment_data = {
    # 1. 實驗基本資訊
    'execution_script': 'Simulation.py',
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
save_path = os.path.join(save_dir, "simulation_data.npz")
np.savez_compressed(save_path, **experiment_data, allow_pickle=True)
print(f"\n實驗數據已保存至: {save_path}")
print(f"實驗時長: {int(experiment_duration//60)}分{int(experiment_duration%60)}秒")

#保存GIF和繪製誤差
#PlotExporter.save_mp4()
PlotExporter.plot_error(data_collector['error_list'])