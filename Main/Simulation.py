import os
import time
import warnings
from datetime import datetime

import control as ctrl
import numpy as np
import scipy
import torch

import CNC
from PPO_brain import PPO

np.set_printoptions(precision=5, suppress=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
time_start = time.time()

# ============================================================================
#                          操作區（每次執行前調整）
# ============================================================================
#region
# --- 模型 ---
read = True
use_switch_model = True            # True: 雙模型切換(追蹤+共振), False: 單模型
if use_switch_model:
    read_file_name1 = 'ModelBUE1.pth'   # 追蹤模型
    read_file_name2 = 'ModelPRE1.pth'   # 共振模型
else:
    read_file_name  = 'ModelPRE1.pth'   # 單模型

# --- 中途切換受控體 Plant：(切換秒數, CNCModel方法名)；自動補到同階數，X0 可安全帶過切換點 ---
# 範例：第 3 秒從 ID_Plant 切到 test_Plant → [(0.0, 'ID_Plant'), (3.0, 'test_Plant')]
PLANT_SCHEDULE = [
    (0.0, 'ID_Plant'),
    (3.0, 'test_Plant'),
]
#endregion

# ============================================================================
#                          固定參數（一般勿動）
# ============================================================================
#region
Ts   = 0.001                        # 取樣時間 (s)【全域固定】
pdl  = 300                          # 路徑區段長度 (samples)，300ms/區間【全域固定】
n_states = 131                      # 狀態維度 action(28)+path_FFT(100)+resonance(2)+error(1)【與 .pth 綁定】
numFC    = 14                       # 頻率限制點數量【與 .pth 綁定】
bound = 20 * np.log10(3000)         # Actor 輸出上下界 (dB)
x_polegain = 0.4352                 # X軸極點縮放係數
z_polegain = 0.4952                 # Z軸極點縮放係數

# --- 共振偵測 / 發散判斷 ---
fft_limit_freq  = 15                # path_FFT 頻率上限 (Hz)
num_low_freq_FC = 3                 # 低頻限制點數量
max_error_um    = 10000             # 最大容許誤差 (um)，超過視為發散

# --- QCQP reward 權重（本檔專用，與 Training/Runtime 刻意不同，勿連動改）---
class CNC_parameter:
    Lq           = 10               # Q 參數階數
    w_sumError   = 1e+3             # 誤差權重
    w_FCfreq     = 1e+0             # FC 分布均勻度權重
    w_Wgc        = 1e+3             # Wgc 懲罰權重 (semiSolved)
    w_earlyTrain = 5e-3             # Infeasible 懲罰權重

# PPO 超參數（模擬不訓練，僅建立網路用）
class PPO_parameter:
    n_step_learning = 20
    mini_batch = 30
    batch_size = 2000
    n_round_batch = 60
    gamma = 0.9
    epsilon = 0.03
    c_update_steps = 10
    a_update_steps = 3
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
def _pad_tf_order(tf, target_den_deg):
    """把離散 TF 乘 z^k 補到指定分母階數：H(z) 不變、階數升高（分子分母同乘 z^k）。"""
    num = list(np.atleast_1d(tf.num[0][0]).astype(float))
    den = list(np.atleast_1d(tf.den[0][0]).astype(float))
    k = target_den_deg - (len(den) - 1)
    if k > 0:
        num = num + [0.0] * k
        den = den + [0.0] * k
    return ctrl.TransferFunction(num, den, tf.dt)
def get_phase_index(step):
    """依目前 step 落在哪個切換點回傳段索引。"""
    idx = 0
    for i, sw in enumerate(_switch_steps):
        if step >= sw:
            idx = i
    return idx
def generate_controller(state, target_path_index, error_input, generation_step):
    """產生控制器：generation_step 表示「模型決策/控制器計算」發生的 step。"""
    global switch, switch_step

    if use_switch_model:
        if resonance_detected >= 1:
            if switch == False:
                print("Switch Model : step ", generation_step, "\n")
                switch = True
                switch_step = generation_step
            action_db = np.array(NO2agent.choose_action(state))
            generated_model_tag = 'NO2_resonance'
        else:
            action_db = np.array(NO1agent.choose_action(state))
            generated_model_tag = 'NO1_tracking'
    else:
        action_db = np.array(agent.choose_action(state))
        generated_model_tag = 'model1'

    action = 10 ** (action_db / 20)
    generated_FC = np.zeros((numFC, 2))
    generated_FC[:, 0] = action[:numFC]
    generated_FC[:, 1] = action[numFC:]
    generated_FC = generated_FC[np.argsort(generated_FC[:, 0])]

    generated_status, generated_CC, _, generated_manual_FC = costfunction_x.switch_controller(
        path, target_path_index, generated_FC.copy(), error_input
    )

    return (
        generated_status,
        generated_CC.copy(),
        generated_FC.copy(),
        generated_manual_FC.copy(),
        generated_model_tag,
    )
#endregion

# ============================================================================
#                             創造實例
# ============================================================================
#region
if use_switch_model:
    NO1agent = PPO(n_states, numFC * 2, bound, PPO_parameter, device)  # 追蹤模型
    NO2agent = PPO(n_states, numFC * 2, bound, PPO_parameter, device)  # 共振模型
else:
    agent = PPO(n_states, numFC * 2, bound, PPO_parameter, device)

model_x = CNC.CNCModel('x', Ts)
path_model = CNC.PathModel(Ts)
ID_Plant = model_x.ID_Plant()
testpath = path_model.test_path()
testpath2 = path_model.test_path2()
up_down_chirp = path_model.up_down_chirp()

costfunction_x = CNC.Costfunction(CNC_parameter, x_polegain, ID_Plant, testpath, pdl, numFC, num_low_freq_FC)
PlotExporter = CNC.PlotExporter()
#endregion

# ============================================================================
#                             讀取資料
# ============================================================================
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

# ============================================================================
#                            預設控制器
# ============================================================================
#region
#邵平控制器
CCnub = [8885.431062041,       -24637.015400412543,     31725.764043065406,
 -24272.73172976235,      10722.74306304059  ,    -2552.3219619078864,
    745.8723434910148,     -332.2659082659679,        1.,
     -1.344771898714865,      0.551909622794037  ,   -0.040436304715828,
     -0.093678503305545,     -0.018232768906639  ,   -0.011387297705785,
     -0.]
CC_shaoping = ctrl.tf2ss(ctrl.TransferFunction(CCnub[:8], CCnub[8:], Ts))
#endregion

# ============================================================================
#                             實驗開始
# ============================================================================
#region
FC = np.zeros((numFC, 2))

# ---- 依 PLANT_SCHEDULE 建立各段 plant，並補到相同階數（維持 X0 維度）----
_schedule = sorted(PLANT_SCHEDULE, key=lambda x: x[0])          # 依切換時間排序
_phase_names = [name for _, name in _schedule]
_phase_plants = [getattr(model_x, name)() for name in _phase_names]  # 各段 plant dict（建一次固定）
_max_den_deg = max(len(p['v2p'].den[0][0]) - 1 for p in _phase_plants)
for p in _phase_plants:                                          # 全部補到最高階
    p['v2p'] = _pad_tf_order(p['v2p'], _max_den_deg)
_switch_steps = [int(round(t / (pdl * Ts))) for t, _ in _schedule]  # 秒 → step

print(f"[Plant 排程] {[(t, n) for (t, _), n in zip(_schedule, _phase_names)]}，"
      f"各段補到 {_max_den_deg} 階")
Plant = _phase_plants[0]      # 初始段
_current_phase = 0

path = testpath
path_index = 0
num_segments = int((len(path) - path_index) / pdl)
dominant_freq = 0.1
if use_switch_model:
    switch = False
    switch_step = None
    resonance_detected = 0
#暫存區歸零
costfunction_x.initialize()
data_buffer, ek_buffer, X0, _, error_history = reset_episode_buffers(num_segments, pdl)
ek = np.zeros(pdl)

#創建實驗資料收集器
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

# step 0 前只產生 C0
zero_error = np.zeros(pdl)
path_FFT_magnitude, dominant_freq, _ = path_FFT(path, 0, dominant_freq)
last_solved_FC = costfunction_x.last_solved_FC
initial_state = np.concatenate([
    state_action(last_solved_FC),
    path_FFT_magnitude,
    np.zeros(3),
])
apply_packet = generate_controller(initial_state, 0, zero_error, -1)

# step k 開始時可取得的是 e(k-1)；step 0 使用握手零誤差
available_error = zero_error.copy()
available_resonance_state = [0, 0]

# 主迴圈：step k 先用 e(k-1) 產生 C(k+1)，再把既有 Ck 套用到 segk
try:
    for step in range(num_segments):
        status, CC, applied_FC, applied_manual_FC, model_tag = apply_packet

        # 本 step 產生下一個控制器；它到下一個 segment 才會被套用
        next_packet = None
        generated_manual_FC = np.empty((0, 2))
        # model_used 記錄本 step 用來產生下一個控制器的模型，
        # 不是本 step 實際套用控制器的來源模型。
        generated_model_tag = model_tag
        if step + 1 < num_segments:
            if use_switch_model and available_resonance_state[0] != 0:
                resonance_detected += 1

            target_step = step + 1
            target_path_index = target_step * pdl
            path_FFT_magnitude, dominant_freq, _ = path_FFT(
                path, target_path_index, dominant_freq
            )
            next_state = np.concatenate([
                state_action(applied_FC),
                path_FFT_magnitude,
                available_resonance_state,
                state_error(available_error),
            ])
            next_packet = generate_controller(
                next_state, target_path_index, available_error, step
            )
            generated_manual_FC = next_packet[3].copy()
            generated_model_tag = next_packet[4]

        path_index = step * pdl
        path_segment = path[path_index:path_index + pdl]

        # 依排程選當前段的 plant（同階數，X0 可安全帶過切換點）
        phase = get_phase_index(step)
        if phase != _current_phase:
            print(f">>> 切換 Plant @ step {step} ({step*pdl*Ts:.2f}s): "
                  f"{_phase_names[_current_phase]} → {_phase_names[phase]}")
            _current_phase = phase
        Plant = _phase_plants[phase]

        # 真正執行的是上一個 step 已經完成的 Ck，不是本 step 剛產生的 C(k+1)
        X0, current_error, _ = CNC.SimulateResponse(
            path_segment.copy(), CC.copy(), Plant['v2p'], X0, Ts
        )
        PlotExporter.plot_frame(CC, ID_Plant['v2p'], applied_FC, applied_manual_FC)

        # 共振資料與當步實際誤差、實際套用控制器保持一致
        resonance_state = state_max_resonance(
            CC, ID_Plant["v2p"], Ts, path_segment, current_error
        )
        freq_linear = 10 ** ((resonance_state[0] / 20) * 30) if resonance_state[0] != 0 else 0
        mag_dB = resonance_state[1] * 30

        data_collector['CC_list'].append(CC.copy())
        data_collector['FC_list'].append(applied_FC.copy())
        # Plot 的定義：manual_FC_list[step] 是本 step 新增的限制，控制器在 step+1 套用
        data_collector['manual_FC_list'].append(generated_manual_FC.copy())
        data_collector['error_list'].append(current_error.copy())
        data_collector['status_list'].append(status)
        data_collector['model_used'].append(generated_model_tag)
        data_collector['resonance_freq_list'].append(freq_linear)
        data_collector['resonance_gain_list'].append(mag_dB)
        print(f"{step:4d} | {status:<11} | {freq_linear:10.5g} | {mag_dB:12.5g}")

        # 判斷是否發散
        if np.sqrt(np.mean(current_error**2)) > max_error_um:
            num_segments = step + 1
            break

        # e_step 只能在下一個 step 才被模型使用
        available_error = current_error.copy()
        available_resonance_state = resonance_state

        # 本 step 產生的 C(k+1)，下一個 step 才成為實際套用控制器
        if next_packet is not None:
            apply_packet = next_packet
except KeyboardInterrupt:
    print("\n[INFO] 使用者中止 (Ctrl+C)，仍會存下已收集的資料...")
except Exception as e:
    import traceback
    print(f"[ERROR] 主迴圈異常中止: {e}，仍會存下已收集的資料...")
    traceback.print_exc()


#計算實驗時長
time_end = time.time()
experiment_duration = time_end - time_start

#整理完整實驗數據
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
experiment_data['switch_step'] = None if not use_switch_model else switch_step

#保存數據到 PlotExporter 建立的資料夾（存檔本身也保護，避免半途失敗）
try:
    save_dir = PlotExporter.get_experiment_folder()
    save_path = os.path.join(save_dir, "simulation_data.npz")
    np.savez_compressed(save_path, **experiment_data, allow_pickle=True)
    print(f"\n實驗數據已保存至: {save_path}")
    print(f"實驗時長: {int(experiment_duration//60)}分{int(experiment_duration%60)}秒")
except Exception as e:
    print(f"[ERROR] 存檔失敗: {e}")

#保存MP4和繪製誤差（失敗不影響已存的資料）
try:
    PlotExporter.save_mp4()
    PlotExporter.plot_error(data_collector['error_list'])
except Exception as e:
    print(f"[WARNING] MP4/誤差圖產生失敗（資料已存）: {e}")
#endregion