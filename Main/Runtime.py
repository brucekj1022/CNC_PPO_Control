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

np.set_printoptions(precision=10, suppress=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
time_start = time.time()

# ============================================================================
#                              存取資料
# ============================================================================
#region
read=True
use_switch_model = True  # True: 雙模型切換, False: 單模型
if use_switch_model:
    read_file_name1='ModelBUE1.pth'  # 追蹤模型
    read_file_name2='ModelPRE1.pth'  # 共振模型
else:
    read_file_name='ModelBUE1.pth'
#endregion

# ============================================================================
#                              參數區域
# ============================================================================
#region
# === 模擬參數 ===
Ts = 0.001              # 取樣時間 (s)
pdl = 300               # 路徑區段長度 (samples)，即 300ms 一個區間

# === 神經網路狀態/動作 ===
n_states = 131          # 狀態維度: action(28) + path_FFT(100) + resonance(2) + error(1)
numFC = 14              # 頻率限制點數量
bound = 20 * np.log10(3000)  # Actor 輸出上下界 (dB)

# === PPO 超參數 ===
class PPO_parameter:
    n_step_learning = 20    # N-step 學習步數
    mini_batch = 30         # Mini-batch 大小
    batch_size = 2000       # Replay buffer 大小
    n_round_batch = 60      # 每輪最大 batch 數
    gamma = 0.9             # 折扣因子
    epsilon = 0.03          # PPO clip 範圍
    c_update_steps = 10     # Critic 更新次數
    a_update_steps = 3      # Actor 更新次數

# === QCQP 控制器參數 ===
class CNC_parameter:
    Lq = 10                 # Q 參數階數
    w_sumError = 1e+2       # 誤差權重
    w_FCfreq = 1e+0         # FC 分布均勻度權重
    w_Wgc = 1e+3            # Wgc 懲罰權重 (semiSolved)
    w_earlyTrain = 5e-3     # Infeasible 懲罰權重

x_polegain = 0.4352         # X軸極點縮放係數
z_polegain = 0.4952         # Z軸極點縮放係數

# === FFT 參數 ===
fft_limit_freq = 2          # path_FFT 頻率上限 (Hz)
num_low_freq_FC = 3         # 低頻限制點數量

# === TCP 參數 ===
HOST = "0.0.0.0"
PORT = 5005
#endregion

# ============================================================================
#                              函數區
# ============================================================================
#region
def show_elapsed_time(start_time, end_time):
    """顯示經過時間 (時:分:秒)。"""
    total_seconds = int(end_time - start_time)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    print(f"{hours}:{minutes:02d}:{seconds:02d}")
def path_FFT(path, path_index, prev_dominant_freq):
    """
    計算路徑的 FFT 頻譜，回傳正規化振幅、主頻率、遮罩長度。
    遮罩長度根據上一步主頻率動態調整，確保涵蓋完整週期。
    """
    min_freq = 0.2  # 避免太低頻讓 mask 長度過長
    dominant_freq = max(prev_dominant_freq, min_freq)
    fft_mask_len = int(1 / dominant_freq / Ts * 2)
    N = int(100000 / fft_limit_freq)  # 輸出 100 點 (改 50000 則為 50 點)

    # 取得 FFT 區段
    if path_index + fft_mask_len > len(path):
        path_segment = path[-fft_mask_len:]
    else:
        path_segment = path[path_index:path_index + fft_mask_len]

    # 去除 DC 並加窗
    path_segment = path_segment - np.mean(path_segment)
    window = np.hanning(fft_mask_len)
    windowed_path = window * path_segment
    
    # FFT 計算
    yf = scipy.fft.fft(windowed_path, N)
    xf = scipy.fft.fftfreq(N, Ts)

    # 取正頻部分並限制頻率範圍
    magnitude = np.abs(yf[:N // 2]) / pdl
    xf = xf[:N // 2]
    freq_mask = xf < fft_limit_freq
    magnitude = magnitude[freq_mask]
    xf = xf[freq_mask]
    
    # 正規化
    mag_min, mag_max = np.min(magnitude), np.max(magnitude)
    normalized_mag = (magnitude - mag_min) / (mag_max - mag_min)
    
    # 找主頻率 (峰值 > 0.7 的最低頻率，或最大振幅頻率)
    peaks, _ = scipy.signal.find_peaks(normalized_mag)
    peak_mags = normalized_mag[peaks]
    valid_peaks = peaks[peak_mags >= 0.7]
    
    if len(valid_peaks) > 0:
        dominant_freq = xf[valid_peaks].min()
    else:
        dominant_freq = xf[np.argmax(normalized_mag)]
    
    return normalized_mag, dominant_freq, fft_mask_len
def state_action(FC):
    """將 FC (頻率, 增益) 轉換為 dB 正規化狀態向量。"""
    return 20 * np.log10(np.hstack((FC[:, 0], FC[:, 1])).ravel()) / 30
def state_max_resonance(CC, plant, Ts, path_segment, ek):
    """提取最大共振點的頻率與增益作為狀態 (正規化 dB)。"""
    resonance_freqs, resonance_mags, resonance_gains = CNC.find_resonance(
        CC, plant, Ts, path_segment, ek
    )
    if len(resonance_freqs) > 0:
        idx = np.argmax(resonance_mags)
        freq_dB = 20 * np.log10(resonance_freqs[idx]) / 30
        gain_dB = resonance_gains[idx] / 30
    else:
        freq_dB, gain_dB = 0, 0
    return [freq_dB, gain_dB]
def state_error(ek):
    """計算加權誤差的對數作為狀態。"""
    weights = 0.7 ** np.arange(len(ek))
    sum_error = np.sum(np.abs(ek) * weights)
    return [np.log1p(sum_error) / 10]
def reset_episode_buffers(num_segments, segment_len):
    """重置每輪的暫存區。"""
    data_buffer = [None] * num_segments  # 儲存計算 reward 前的資料
    ek_buffer = np.zeros((3, segment_len))  # 模擬誤差延遲 (3 步緩衝)
    X0 = 0  # 初始狀態
    episode_reward = 0  # 累計 reward
    error_history = []  # 儲存每步誤差
    return data_buffer, ek_buffer, X0, episode_reward, error_history
#endregion

# ============================================================================
#                             創造實例
# ============================================================================
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

# ============================================================================
#                             讀取資料
# ============================================================================
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

# ============================================================================
#                            固定控制器
# ============================================================================
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

#X軸結構共振測試控制器
X_resonance_old=[1.24776671e+04, -2.82535941e+04, 2.20793412e+04, -6.01791813e+03,
           -2.36084310e-02, 2.39126305e-01, -5.14901653e-01, 3.92582295e+03,
           -9.47329155e+03, 7.77157149e+03, -2.22327578e+03, -5.28287522e-01,
           -1.99142081e-02, 1.00000000e+00, -1.36936611e+00, 4.47522069e-01,
           -1.07366429e-01, -3.20512604e-07, -6.61145311e-07, -2.30769282e-06,
           -5.56249046e-06, -2.68773333e-02, -7.62712047e-02, -2.07301160e-05,
           -6.82492589e-07, 6.03972590e-23]
CC_X_resonance_old = ctrl.tf2ss(ctrl.TransferFunction(X_resonance_old[:13], X_resonance_old  [13:], Ts))

# ============================================================================
#                             實驗開始
# ============================================================================
FC = np.zeros((numFC, 2))
data_collector = {
    'CC_list': [],
    'FC_list': [],
    'manual_FC_list': [],
    'error_list': [],
    'status_list': [],
    'resonance_freq_list': [],
    'resonance_gain_list': [],
    'model_used': []
}

path = testpath
path_index = 0
num_segments = int((len(path) - path_index) / pdl)
dominant_freq = 0.1
if use_switch_model:
    switch = False
    resonance_detected = 0
#暫存區歸零
costfunction_x.initialize()
data_buffer, ek_buffer, X0, _, error_history = reset_episode_buffers(num_segments, pdl)
ek = np.zeros(pdl)
CC = None
#準備第一個state
path_FFT_magnitude, dominant_freq, _ = path_FFT(path, path_index, dominant_freq)
last_solved_FC = costfunction_x.last_solved_FC
s = np.concatenate([state_action(last_solved_FC), path_FFT_magnitude, np.zeros(3)])
print("Standby for link in \n")

# 主迴圈用 try 包住：不論正常 break、例外、或 Ctrl+C，都會落到後面的存檔區
try:
    for step in range(num_segments + 1):  # 因為 Error 有延遲所以要多一步收集資料
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
        if step >= 1:
            data_collector['error_list'].append(ek.copy())
            if step == num_segments:
                break
            path_FFT_magnitude, dominant_freq, _ = path_FFT(path, path_index, dominant_freq)
            s = np.concatenate([
                state_action(FC), path_FFT_magnitude,
                state_max_resonance(CC, ID_Plant["v2p"], Ts, path_segment, ek),
                state_error(ek)
            ])

            #雙模型模式：檢測共振換model
            if use_switch_model:
                print(state_max_resonance(CC, ID_Plant["v2p"], Ts, path_segment, ek)[0])
                if state_max_resonance(CC, ID_Plant["v2p"], Ts, path_segment, ek)[0] != 0:
                    resonance_detected += 1

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
        FC = FC[np.argsort(FC[:, 0])]             # 按頻率排序

        #合成新控制器
        status, CC, ek_hat, manual_add_FC = costfunction_x.switch_controller(path, path_index, FC.copy(), ek)
        path_segment = path[path_index:path_index + pdl]

        #使用固定控制器 (可選: CC_shaoping, CC_X_central, CC_X_resonance_old, CC_X_resonance_g6_w800, CC_X_boost_6db)
        # CC = CC_shaoping

        #收集實驗數據（每步）
        data_collector['CC_list'].append(CC.copy())
        data_collector['FC_list'].append(FC.copy())
        data_collector['manual_FC_list'].append(manual_add_FC.copy())
        data_collector['status_list'].append(status)

        #轉出CC
        CC_tf = ctrl.ss2tf(CC)
        den = np.array(CC_tf.den[0][0])
        num = np.array(CC_tf.num[0][0])
        if len(num) < len(den):
            num = np.pad(num, (0, len(den) - len(num)))
        CCdata = np.concatenate((num, den))

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
        path_index += pdl

        #儲存共振資料並打印
        resonance_state = state_max_resonance(CC, ID_Plant["v2p"], Ts, path_segment, ek)
        freq_normalized = resonance_state[0]
        mag_normalized = resonance_state[1]
        freq_linear = 10 ** ((freq_normalized / 20) * 30) if freq_normalized != 0 else 0
        mag_dB = mag_normalized * 30
        data_collector['resonance_freq_list'].append(freq_linear)
        data_collector['resonance_gain_list'].append(mag_dB)
        print(
            f"{step:4d} | "
            f"{status:<11} | "
            f"{freq_linear:10.5g} | "
            f"{mag_dB:12.5g}"
        )
except KeyboardInterrupt:
    print("\n[INFO] 使用者中止 (Ctrl+C)，仍會存下已收集的資料...")
except Exception as e:
    import traceback
    print(f"[ERROR] 主迴圈異常中止: {e}，仍會存下已收集的資料...")
    traceback.print_exc()
 
#關閉server
try:
    conn.close()
except:
    pass

#計算實驗時長
time_end = time.time()
experiment_duration = time_end - time_start

#整理完整實驗數據
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
    'num_segments': num_segments,
    
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

#保存數據到 PlotExporter 建立的資料夾（存檔本身也保護，避免半途失敗）
try:
    save_dir = PlotExporter.get_experiment_folder()
    save_path = os.path.join(save_dir, "runtime_data.npz")
    np.savez_compressed(save_path, **experiment_data, allow_pickle=True)
    print(f"\n實驗數據已保存至: {save_path}")
    print(f"實驗時長: {int(experiment_duration//60)}分{int(experiment_duration%60)}秒")
except Exception as e:
    print(f"[ERROR] 存檔失敗: {e}")

#畫出誤差（畫圖失敗不影響已存的資料）
try:
    PlotExporter.plot_error(data_collector['error_list'])
except Exception as e:
    print(f"[WARNING] 誤差圖繪製失敗（資料已存）: {e}")
