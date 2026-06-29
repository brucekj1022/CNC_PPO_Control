import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch

import CNC
from PPO_brain import PPO, ReplayBuffer

np.set_printoptions(precision=15,suppress=True)#設置打印位數，科學記號
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
time_start=time.time()

# ============================================================================
#                              存取設定
# ============================================================================
#region
read = True
read_file_name = 'Model.pth'
save = True
save_file_name = 'Model.pth'
#endregion

# ============================================================================
#                              參數區域
# ============================================================================
#region
# === 模擬參數 ===
Ts = 0.001              # 取樣時間 (s)
pdl = 300               # 路徑區段長度 (samples)，即 300ms 一個區間
max_error_um = 10000    # 最大容許誤差 (um)，超過視為發散

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

# === 學習率排程 ===
# 格式: [(學習率, 持續輪數), ...]
lr_schedule = [
    (1e-5, 3000),  # 前 1000 輪用 1e-5
    (1e-6, 6000),  # 接下來 2000 輪用 1e-6
    (1e-7, 6000),
]
total_iterations = sum(rounds for _, rounds in lr_schedule)  # 總訓練輪數

# === QCQP 控制器參數 ===
class CNC_parameter:
    Lq = 10                 # Q 參數階數
    w_sumError = 1e+3       # 誤差權重
    w_FCfreq = 4e+3         # FC 分布均勻度權重
    w_Wgc = 1e+3            # Wgc 懲罰權重 (semiSolved)
    w_earlyTrain = 5e-3     # Infeasible 懲罰權重

x_polegain = 0.4352         # X軸極點縮放係數
z_polegain = 0.4952         # Z軸極點縮放係數

# === FFT 參數 ===
fft_limit_freq = 15         # path_FFT 頻率上限 (Hz)
num_low_freq_FC = 3         # 低頻限制點數量

# === 繪圖設定 ===
enable_plot = False         # True: 輸出 Bode 圖/MP4/誤差圖

# === 手動 FC 初始值 (頻率 rad/s, 增益) ===
manual_FC = np.array([
    [0.1,   1000],
    [1,     100],
    [10,    10],
    [100,   1.1],
    [300,   0.8],
    [500,   0.39],
    [700,   0.16],
    [900,   0.07],
    [1000,  0.08],
    [1300,  0.1],
    [1500,  0.09],
    [2000,  0.03],
    [2500,  0.07],
    [3000,  0.1]
])
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
    """重置每輪訓練的暫存區。"""
    data_buffer = [None] * num_segments  # 儲存計算 reward 前的資料
    ek_buffer = np.zeros((3, segment_len))  # 模擬誤差延遲 (3 步緩衝)
    X0 = 0  # 初始狀態
    episode_reward = 0  # 累計 reward
    error_history = []  # 儲存每步誤差
    return data_buffer, ek_buffer, X0, episode_reward, error_history
def get_lr_for_iteration(total_iter):
    """根據累計輪數回傳對應的學習率。"""
    cumulative = 0
    for lr, rounds in lr_schedule:
        cumulative += rounds
        if total_iter <= cumulative:
            return lr
    return lr_schedule[-1][0]  # 超過排程範圍則用最後一個
#endregion

# ============================================================================
#                             創造實例
# ============================================================================
#region
agent = PPO(n_states , numFC*2, bound, PPO_parameter, device)
replay_buffer=ReplayBuffer(PPO_parameter.batch_size)

model_x = CNC.CNCModel('x',Ts)#創建馬達實例
path_model=CNC.PathModel(Ts)
ID_Plant=model_x.ID_Plant()#取得馬達ID模型
training_path=path_model.training_path()
testpath=path_model.test_path()
testpath2=path_model.test_path2()

costfunction_x=CNC.Costfunction(CNC_parameter, x_polegain, ID_Plant, testpath, pdl, numFC, num_low_freq_FC)
PlotExporter = CNC.PlotExporter() if enable_plot else None
#endregion

# ============================================================================
#                             讀取資料
# ============================================================================
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

# ============================================================================
#                             訓練本體
# ============================================================================
#region
all_iter_r = []
FC = np.zeros((numFC, 2))
current_lr = None  # 記錄當前學習率，用於偵測切換
#endregion

for iteration in range(1, total_iterations + 1):
    # 檢查是否需要切換學習率（基於累計輪數）
    current_total_iter = start_iteration + iteration
    lr = get_lr_for_iteration(current_total_iter)
    if lr != current_lr:
        current_lr = lr
        agent.set_learning_rate(lr)
        print(f">>> 累計 {current_total_iter} 輪，學習率切換為 {lr}")
    
    # 選擇模型：高學習率用 ID 模型，低學習率用隨機模型
    if current_lr >= 1e-5:
        Plant = model_x.ID_Plant()
    else:
        Plant = model_x.PRE_Plant()
    
    # 每輪初始狀態
    path = training_path[iteration % len(training_path)]
    path_index = (iteration - 1) % pdl
    num_segments = int((len(path) - path_index) / pdl)
    dominant_freq = 0.1  # 初始 FFT 遮罩頻率
    
    # 暫存區歸零
    costfunction_x.initialize()
    data_buffer, ek_buffer, X0, episode_reward, error_history = reset_episode_buffers(num_segments, pdl)
    # 準備第一個 state
    path_FFT_mag, dominant_freq, _ = path_FFT(path, path_index, dominant_freq)
    last_solved_FC = costfunction_x.last_solved_FC
    s = np.concatenate([state_action(last_solved_FC), path_FFT_mag, np.zeros(3)])
    
    for step in range(num_segments + 1):  # +1 因為誤差有延遲
        # 檢查是否發散
        rms_error = np.sqrt(np.mean(ek_buffer[step % 3] ** 2))
        if rms_error > max_error_um:
            num_segments = step - 1
            break
        if step >= num_segments:
            continue
        
        # 產生動作 (Actor 輸出 dB → 轉換為線性值)
        a = np.array(agent.choose_action(s))
        action_linear = 10.0 ** (a / 20.0)
        FC[:, 0] = action_linear[:numFC]  # 頻率
        FC[:, 1] = action_linear[numFC:]  # 增益
        FC = FC[np.argsort(FC[:, 0])]     # 按頻率排序
        
        # 合成控制器並模擬
        status, CC, ek_hat, manual_add_FC = costfunction_x.switch_controller(path, path_index, FC.copy(), ek_buffer[step % 3])
        path_segment = path[path_index:path_index + pdl]
        X0, ek_buffer[(step + 2) % 3, :], _ = CNC.SimulateResponse(path_segment.copy(), CC.copy(), Plant['v2p'], X0, Ts)
        if enable_plot:
            PlotExporter.plot_frame(CC, Plant['v2p'], FC, manual_add_FC)

        # 計算下一步狀態並存入 buffer
        path_FFT_mag, dominant_freq, _ = path_FFT(path, path_index + pdl, dominant_freq)
        s_ = np.concatenate([
            state_action(FC),
            path_FFT_mag,
            state_max_resonance(CC, ID_Plant["v2p"], Ts, path_segment, ek_buffer[(step + 1) % 3]),
            state_error(ek_buffer[(step + 1) % 3])
        ])
        data_buffer[step] = (
            s.copy(), a.copy(), s_.copy(), FC.copy(), 
            status, CC.copy(), path_segment.copy(), ek_buffer[(step + 2) % 3, :].copy()
        )
        
        # 準備下一步
        s = s_
        path_index += pdl

    # 從 data_buffer 計算 reward 並放進 replay_buffer
    for i in range(num_segments):
        s, a, s_, FC, status, CC, path_segment, ek = data_buffer[i]
        error_history.append(ek.copy())
        r = costfunction_x.reward(FC, status, CC, ek, visual=0)
        episode_reward += r
        replay_buffer.push(s.copy(), a.copy(), r, s_.copy())
    
    print(f"Iteration: {iteration:<5} Reward: {episode_reward:.2f}\n")
    if enable_plot:
        PlotExporter.save_mp4()
        PlotExporter.plot_error(error_history)

    # 訓練 PPO
    agent.training(replay_buffer)
    
    # 每 100 輪儲存模型
    if iteration % 100 == 0:
        save_path = f'../Model/{save_file_name}'
        model_dict = torch.load(save_path) if os.path.exists(save_path) else {}
        key = f'iteration:{iteration + start_iteration}'
        model_dict[key] = {
            'reward': all_iter_r[-100:],
            'actor': agent.actor_model.state_dict(),
            'critic': agent.critic_model.state_dict(),
            'FC': costfunction_x.last_solved_FC
        }
        torch.save(model_dict, save_path)
    
    # 平滑後存入列表
    smoothed_reward = episode_reward if iteration == 1 else all_iter_r[-1] * 0.9 + episode_reward * 0.1
    all_iter_r.append(smoothed_reward)

# 訓練結束
time_finish = time.time()
plt.title("Return")
plt.plot(np.arange(len(all_iter_r)), all_iter_r)
plt.show()
show_elapsed_time(time_start, time_finish)