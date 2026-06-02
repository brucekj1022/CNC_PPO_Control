import argparse
import os
import time
import warnings

import control as ctrl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch

import CNC
from PPO_brain import PPO, ReplayBuffer

np.set_printoptions(precision=15,suppress=True)#設置打印位數，科學記號
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
time_start=time.time()

######   存取資料   ######
#region
read=True
read_file_name='Model.pth'
save=True
save_file_name='Model.pth'
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
parser2.add_argument('--w_FCfreq', type=float, default=4e+3)
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
agent = PPO(n_states , numFC*2, bound, PPO_parameter, device)
replay_buffer=ReplayBuffer(PPO_parameter.batch_size)

model_x = CNC.CNCModel('x',Ts)#創建馬達實例
path_model=CNC.PathModel(Ts)
ID_Plant=model_x.ID_Plant()#取得馬達ID模型
training_path=path_model.training_path()
testpath=path_model.test_path()
testpath2=path_model.test_path2()

costfunction_x=CNC.Costfunction(CNC_parameter, x_polegain, ID_Plant, testpath, pdl, numFC, num_low_freq_FC)#manual_FC.copy()
PlotExporter=CNC.PlotExporter()
#endregion

#####   讀取資料   ####
 #region
if(read==True):
    try:
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        checkpoint = torch.load(f"../Model/{read_file_name}",map_location="cpu",weights_only=False)  # 僅在可信來源使用
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
            status, _,_,_=costfunction_x.switch_controller(testpath,0, checkpoint[iteration_key]['FC'], np.array([0]))
            costfunction_x.set_controller(checkpoint[iteration_key]['FC'])
            print(f"載入model {read_file_name} 第{iteration_key}輪，初始限制條件{status}")
    except FileNotFoundError:
        start_iteration =0
        print(f"找不到model檔案，從0訓練")
else : 
    start_iteration =0
    print(f"不引進model，從0訓練")
 #endregion

######   訓練本體   ######
#創造空容器
#region
all_iter_r = []
FC = np.zeros((numFC, 2))
#endregion

for iteration in range(1, PPO_parameter.iteration+1):#總輪數
    if PPO_parameter.learning_rate==1e-5 : Plant=model_x.ID_Plant()
    else:Plant=model_x.random_Plant()
    #每一輪的初始狀態
    path=training_path[iteration%len(training_path)]
    path_index=(iteration-1)%pdl
    #path=testpath
    #path_index=0
    num_path_district=int((len(path)-path_index)/pdl)
    prev_dominant_freq=0.1#初始遮罩設定
    #暫存區歸零 
    costfunction_x.initialize()
    data_buffer, ek_buffer, X0, iter_r, error = reset_buffers(num_path_district, pdl)
    #準備第一個state
    path_FFT_magnitude, prev_dominant_freq, _=path_FFT(path, path_index , prev_dominant_freq)
    last_solved_FC=costfunction_x.last_solved_FC
    s=np.concatenate([state_action(last_solved_FC), path_FFT_magnitude, np.zeros(3)])#合成第一組狀態
    for step in range(num_path_district+1):#開始產生歷程，因為Error有延遲所以要多兩步收集資料
        #判斷是否發散
        if np.sqrt(  np.mean(  (ek_buffer[step%3])**2  )  ) > max_error_um:#判斷Error有沒有發散
            num_path_district=step-1
            break
        if(step>=num_path_district):
            continue
        #產生動作
        a = np.array(agent.choose_action(s))      # dB
        action = 10.0 ** (a / 20.0)               # 線性倍率
        FC[:, 0] = action[:numFC]                 # 頻率
        FC[:, 1] = action[numFC:]                 # 增益
        sorted_indices = np.argsort(FC[:, 0])  #排序FC的索引
        FC = FC[sorted_indices]  # 按照索引排序FC
        #是否合成新控制器並模擬運行
        status, CC, ek_hat, manual_add_FC=costfunction_x.switch_controller(path, path_index, FC.copy(), ek_buffer[step%3])

        CC_tf = ctrl.ss2tf(CC)
        den = np.array(CC_tf.den[0][0], dtype=np.float32)
        cdl=len(den)#controller_data_len
        num = np.array(CC_tf.num[0][0], dtype=np.float32)
        num = np.pad(num, (0, cdl - len(num)), mode='constant')#補齊避免分子階數不足
        print(den,num)

        path_district=path[path_index : path_index+pdl]
        X0, ek_buffer[(step+2)%3 , :], _=CNC.SimulateResponse(path_district.copy(), CC.copy(), Plant['v2p'], X0,Ts)#模擬輸出Controller給工具機
        #PlotExporter.plot_frame(CC, Plant['v2p'], FC, manual_add_FC)#製作Gif

        #資料存進data_buffer
        path_FFT_magnitude, prev_dominant_freq, _=path_FFT(path, path_index+pdl, prev_dominant_freq)
        s_ = np.concatenate([ state_action(FC), path_FFT_magnitude, state_maxResonance(CC, ID_Plant["v2p"], Ts, path_district, ek_buffer[(step+1)%3]), state_error(ek_buffer[(step+1)%3])])
        data_buffer[step]=(s.copy(), a.copy(), s_.copy(), FC.copy(), status, CC.copy(), path_district.copy(), ek_buffer[(step+2)%3 , :].copy())
        #準備下一步
        s=s_
        path_index=path_index+pdl

    #從data_buffer計算reward並放進replay_buffer
    for i in range(0,num_path_district):
        s, a, s_, FC, status, CC, path_district, ek=data_buffer[i]
        error.append(ek.copy())
        r=costfunction_x.reward(FC, status, CC, ek, visual=0)
        iter_r+=r
        replay_buffer.push(s.copy(), a.copy() , r , s_.copy())
        #print(f"iteration:{iteration:<5}step:{i+1:<3}  status:{status:13}reward:{r:2f}")
    print(f"Iteration:{iteration:<5}Reward:{iter_r:2f}\n")
    #PlotExporter.save_mp4()
    #PlotExporter.plot_error(error)

    #訓練
    agent.training(replay_buffer)
    #儲存model
    if  iteration % 100 == 0:
        save_path=f'../Model/{save_file_name}'
        model_dict = torch.load(save_path) if os.path.exists(save_path) else {}
        key = f'iteration:{iteration+start_iteration}'
        model_dict[key] = {
            'reward': all_iter_r[-100:],
            'actor': agent.actor_model.state_dict(),
            'critic': agent.critic_model.state_dict(),
            'FC':costfunction_x.last_solved_FC
        }
        torch.save(model_dict, save_path)
    #把return存入列表
    all_iter_r.append(iter_r if iteration == 1 else all_iter_r[-1] * 0.9 + iter_r * 0.1)#平滑顯示資料

time_finish=time.time()
plt.title("Return")
plt.plot(np.arange( len(all_iter_r)), all_iter_r)
plt.show()
Time_show(time_start, time_finish)