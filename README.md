# CNC 使用 PPO — 進給軸控制器線上自動設計系統

CNC 進給軸控制器的線上自動設計系統。核心是用 **PPO 強化學習** 訓練一個 Actor，輸出「頻域增益限制點」(FC)，再透過 **QCQP 最佳化**（Youla/Q 參數化 + 互質分解 + 線性分式轉換 LFT，以 Gurobi 求解）即時合成控制器 `CC`，套用到 CNC 馬達受控體上，目標是壓制機台高頻共振同時維持追跡精度。

> 本專案註解、文件、版本紀錄皆為繁體中文。

---

## 目錄

- [控制流程](#控制流程)
- [專案目錄結構](#專案目錄結構)
- [執行方式](#執行方式)
  - [相依套件](#相依套件)
- [Main/ 各檔案說明（用途、輸入、輸出）](#main-各檔案說明用途輸入輸出)
  - [`CNC.py` — 系統核心](#cncpy--系統核心)
  - [`PPO_brain.py` — PPO 演算法](#ppo_brainpy--ppo-演算法)
  - [`Training.py` — 離線 PPO 訓練（啟動檔）](#trainingpy--離線-ppo-訓練啟動檔)
  - [`Simulation.py` — 離線閉迴路模擬](#simulationpy--離線閉迴路模擬)
  - [`Runtime.py` — 上機實時運行](#runtimepy--上機實時運行)
  - [`pc_server.py` — 通訊協議模組 + 測試工具](#pc_serverpy--通訊協議模組--測試工具)
  - [`Toolbox.py` — 工具選單](#toolboxpy--工具選單)
  - [`Plot_Exp_Data.py` — 實驗資料視覺化](#plot_exp_datapy--實驗資料視覺化)
- [Matlab/ 各檔案說明](#matlab-各檔案說明)
- [資料檔案格式](#資料檔案格式)
- [約定與注意事項](#約定與注意事項)

---

## 控制流程

```
State → Actor(PPO) → FC → QCQP(Costfunction) → 控制器 CC → 模擬/上機 → 誤差/共振 → 下一個 State
```

- **State** (131 維)：`action(28) + path_FFT(100) + maxResonance(2) + sumError(1)`
- **Action** (28 維)：`numFC=14` 個限制點 ×（頻率, 增益）。Actor 輸出為 dB，外部轉成線性值。
- **全域固定值**：`pdl=300`（每區段 300 ms）、`Ts=0.001`、`numFC=14`、`bound=20·log10(3000)` dB。
  > ⚠️ 不要更動 State 維度 (131) 或 numFC (14)，否則與已訓練的 `.pth` 不相容。

---

## 專案目錄結構

```
2026.4.24主線/
├── CLAUDE.md              # 給 Claude Code 的專案指引
├── README.md             # 本文件
├── 版本紀錄.md            # 每版模型/參數/流程差異追蹤
├── CNC使用PPO.sln         # Visual Studio 方案檔
│
├── Main/                 # Python 主程式（所有腳本都在此目錄下執行）
│   ├── CNC.py            # 系統核心：受控體模型、QCQP 控制器設計、繪圖輸出
│   ├── PPO_brain.py      # PPO 演算法：Actor/Critic 網路、ReplayBuffer
│   ├── Training.py       # 離線 PPO 訓練（啟動檔）
│   ├── Simulation.py     # 離線閉迴路模擬
│   ├── Runtime.py        # 上機：TCP server 等 LabVIEW/cRIO 連線
│   ├── pc_server.py      # 通訊協議模組 + 獨立 TCP 連線測試工具
│   ├── Toolbox.py        # 工具選單（波德圖、極點、共振、路徑）
│   ├── Plot_Exp_Data.py  # 視覺化實驗 npz（單一/批次統計）
│   ├── Test.py           # 測試用暫存檔（目前為空）
│   ├── Delta_Data.mat    # 不確定性模型集合（由 Matlab 產生，CNC.py 載入）
│   └── CNC使用PPO.pyproj  # Visual Studio Python 專案檔
│
├── Matlab/               # 系統鑑別、模型產生（受控體模型參數來源）
│   ├── Model_Data.m      # 名義模型定義（共用配置，硬編碼 ID 係數）
│   ├── ID_Model.m        # 系統辨識輔助工具（啟動 System ID GUI）
│   ├── Create_Delta.m    # 計算不確定性，產生 Delta_Data.mat
│   ├── Plot_OE_Model.m   # OE 模型精度評估與排名
│   ├── Delta_Data.mat    # 不確定性模型（需複製到 Main/ 供 Python 使用）
│   ├── 2025.9.15 ID data/        # 早期鑑別資料（CSV）
│   ├── 2025.9.17 velocityIO_data/ # 主驗證資料（9 組 Input/Output 速度）
│   ├── OE Model/         # OE 多項式模型（*.mat）與評估結果（*.png/csv）
│   └── Check model/      # 模型驗證
│       ├── CheckModel.m  # 極點/零點/穩定裕度檢驗
│       ├── Cloop_sim.m   # 閉迴路模擬
│       └── CLoopsim.slx  # Simulink 閉迴路模擬框架
│
├── Model/                # 訓練權重 .pth（已 gitignore，太大另存雲端）
└── ExperimentData/       # 實驗資料（已 gitignore，量大）
```

---

## 執行方式

無 build / lint / test 框架（這是 Visual Studio Python 專案，啟動檔為 `Training.py`）。直接用 Python 執行各腳本。

**重要：所有 Python 腳本都必須在 `Main/` 目錄下執行**，因為使用相對路徑：
- `Delta_Data.mat` 從工作目錄載入（`Main/Delta_Data.mat`）
- 模型權重讀寫 `../Model/`，實驗資料寫 `../ExperimentData/`

```bash
cd Main
python Training.py       # 離線 PPO 訓練
python Simulation.py     # 離線閉迴路模擬
python Runtime.py        # 上機（TCP server）
python Toolbox.py        # 工具選單
python pc_server.py      # 獨立 TCP 連線測試
python Plot_Exp_Data.py  # 視覺化實驗資料
```

### 相依套件

無 `requirements.txt`。需要：`numpy`, `scipy`, `control`, `torch`, `gurobipy`, `imageio`, `matplotlib`, `pandas`, `openpyxl`。

- **`gurobipy` 需要有效的 Gurobi 授權** — 控制器合成的 QCQP 求解完全依賴它，沒授權無法跑 `Costfunction`。
- `torch` 自動偵測 CUDA，無 GPU 會 fallback 到 CPU。
- Matlab 腳本需要 **System Identification Toolbox**、**Robust Control Toolbox**（ultidyn）、**Simulink**。

---

## Main/ 各檔案說明（用途、輸入、輸出）

### `CNC.py` — 系統核心

整個系統的核心邏輯都在這個模組，其他腳本只是不同的執行入口。**不會單獨執行**，被其他腳本 import。

**模組層級函數**
- `SimulateResponse(path, CC, plant, X0, Ts)` — 模擬閉迴路響應，回傳 `(下一步狀態, 誤差um, 輸出)`
- `find_Wgc(CC, plant, Ts)` — 用切比雪夫多項式法求增益交越頻率
- `find_resonance(...)` — 從誤差頻譜辨識高頻共振點（頻率 > Wgc 且振幅 > 5×輸出均值）

**`CNCModel`** — 受控體模型（軸別 `'x'`/`'z'`，目前主用 X 軸）。每個方法回傳 `{'v2p':…, 'v2v':…, 'Ts':…}`（v2p=速度→位置，v2v=速度→速度）：
- `ID_Plant()` — 系統鑑別得到的標稱模型（參數來自 MATLAB ID）
- `test_Plant()` — 標稱 + 固定共振點
- `BUE_Plant()` — 從 `Delta_Data.mat` 隨機抽一組不確定性模型（Base Uncertainty Ensemble）
- `PRE_Plant()` — BUE + 隨機高頻共振（Perturbed Resonant Ensemble）

**`PathModel`** — 路徑生成：`test_path`(0~1Hz chirp)、`test_path2`(0~8Hz)、`training_path`(20 條混合)、`up_down_chirp`

**`Costfunction`** — QCQP 控制器設計（最重要的類別）
- `switch_controller(...)` — PPO 每步主要入口：解 QCQP + 共振壓制，回傳 `(status, CC, ek_hat, manual_add_FC)`
- `optimizationcvx()` — 實際的 Gurobi QCQP 求解
- `reward()` — PPO 獎勵函數
- `status` 三態：`"Solved"` / `"semiSolved"` / `"Infeasible"`

**`PlotExporter`** — 在 `../ExperimentData/<時間戳>/` 下產生繪圖。**輸出檔案**：
| 檔案 | 內容 |
|------|------|
| `frames/frame_NNN.png` | 每步 Bode 圖 + FC 點（中間檔） |
| `fft_frames/fft_frame_NNN.png` | 每步誤差 FFT 圖（中間檔） |
| `frequency_response.mp4` | Bode 圖動畫 |
| `error.png` | 誤差波形圖 |
| `experiment_info.txt` | 實驗資訊文字 |

### `PPO_brain.py` — PPO 演算法

`PPO` 類別 + `ActorNet`/`CriticNet`（連續動作，Actor 輸出 mu/sigma）+ `ReplayBuffer`。被各執行腳本 import，不單獨執行。

### `Training.py` — 離線 PPO 訓練（啟動檔）

- **讀取**：`../Model/{read_file_name}`（預設 `ModelBUE1.pth`，可中斷續訓）、`Delta_Data.mat`（經 CNCModel 載入）
- **寫出**：`../Model/{save_file_name}`（預設 `Model.pth`，每 100 輪追加一個 `iteration:N` 鍵）
- 用 `training_path`（20 條混合路徑）訓練。`lr_schedule` 分階段降學習率：**高學習率階段用 `ID_Plant`，低學習率階段切到 `PRE_Plant`（隨機共振）**。
- `w_FCfreq = 4e+3`（FC 均勻度權重高）、`fft_limit_freq = 15`、`enable_plot=False`（預設關閉繪圖加速）

### `Simulation.py` — 離線閉迴路模擬

- **讀取**：`../Model/{read_file_name}`（預設 `ModelPRE1.pth`）；`use_switch_model=True` 時讀兩個模型
- **寫出**：`../ExperimentData/{時間戳}/simulation_data.npz` + Bode 動畫、`error.png`
- 用單一 `test_path`（0~1Hz chirp）模擬。`w_FCfreq = 1e+0`、`fft_limit_freq = 15`

### `Runtime.py` — 上機實時運行

- 結構與 `Simulation.py` 相同，但作為 **TCP server**（預設 `0.0.0.0:5005`）等 LabVIEW/cRIO 連線
- **讀取**：`../Model/{read_file_name}`（預設 `ModelBUE1.pth`）
- **寫出**：`../ExperimentData/{時間戳}/runtime_data.npz` + Bode 動畫、`error.png`
- 每步：收逗號分隔的誤差 `ek`（長度=pdl=300）→ 算新 `CC` → 傳回控制器係數
- `w_sumError = 1e+2`（最低）、`fft_limit_freq = 2`（實時響應快）

### `pc_server.py` — 通訊協議模組 + 測試工具

- 提供 `array_to_str` / `str_to_array` / `recv` / `send` 等通訊函數（供 Runtime.py import）
- **單獨執行**時：當獨立 TCP server，固定回傳一組控制器係數，用於測試連線與量測 RTT。無檔案 I/O。

### `Toolbox.py` — 工具選單

執行後顯示選單，**所有圖預設在螢幕顯示；tool 5/6/7 另存檔到所選實驗資料夾**：
| 編號 | 功能 | 輸出檔（存實驗資料夾） |
|------|------|------|
| 1 | 受控體波德圖（ID/Test/BUE/PRE，可選 v2p/v2v） | 螢幕顯示 |
| 2 | 路徑資料繪圖與匯出（TXT/Excel） | `*.txt` / `*.xlsx` |
| 3 | 隨機共振峰值上下界繪圖（可選 v2p/v2v） | 螢幕顯示 |
| 4 | 動態 FFT 遮罩測試（產生動畫） | `animation.mp4` |
| 5 | 開迴路波德圖（可疊中央控制器） | `openloop_bode.png` |
| 6 | 閉迴路極點圖（含最小阻尼比線） | `closed_loop_poles.png` |
| 7 | 機台共振頻譜分析（互動式時間/濾波/異常值參數） | `error_full_fft.png` |
| 8 | 訓練 Return 曲線（讀 `../Model/*.pth`，可選全部或單一） | 螢幕顯示 |

### `Plot_Exp_Data.py` — 實驗資料視覺化

`BATCH_MODE=True` 批次處理 `BATCH_EXPERIMENTS` 清單；`False` 則彈窗手動選檔/資料夾。

**單實驗（single）模式**，輸出存到該實驗資料夾：
| 檔案 | 內容 |
|------|------|
| `experiment_info.txt` | 實驗基本資訊 + 每步詳細狀況表 |
| `reference_path.png` | 參考路徑時序圖 |
| `error.png` | 誤差時序圖（含事件標記線） |
| `controller_margins.png` | 4 合 1 性能指標：GM / PM / Wgc / 斜率 |
| `frequency_response.mp4` | 每步 Bode 圖 + FC 點動畫 |
| `error_fft.mp4` | 每步誤差 FFT 動畫 |

**多實驗（multi）統計模式**，輸出存到指定資料夾：
| 檔案 | 內容 |
|------|------|
| `statistics_summary.txt` | 統計摘要（實驗數、RMS 統計、各次 RMS） |
| `error_statistics.png` | 誤差均值 + 標準差陰影 |
| `margins_statistics.png` | 4 合 1 性能指標均值 ±1σ |

---

## Matlab/ 各檔案說明

受控體模型參數源自此資料夾。修改 plant 行為通常要回到 MATLAB 重新產生資料。

### 系統辨識與模型開發流程

```
[實驗數據]
  ├─ 2025.9.15 ID data/        ──→ ID_Model.m ──→ System ID GUI ──→ (係數硬編碼進 Model_Data.m)
  └─ 2025.9.17 velocityIO_data/
       ├─ Create_Delta.m ──→ Delta_Data.mat ──→ (複製到 Main/) ──→ CNC.py BUE_Plant()
       └─ Plot_OE_Model.m ──→ OE Model/*.png + OE_Model_RMSE_Result.csv

[模型驗證]
  Model_Data.m ──┬─ CheckModel.m ──→ 極點/零點/穩定裕度
                 └─ Cloop_sim.m ──→ CLoopsim.slx ──→ 閉迴路模擬誤差圖
```

### 腳本

- **`Model_Data.m`** — 名義模型定義（共用配置）。依機台型號建立 `plantX`/`plantZ`（含 v2v、v2p、Ts），硬編碼 ID 係數。無檔案輸出，供其他腳本引用。
- **`ID_Model.m`** — 系統辨識輔助。掃描 `2025.9.15 ID data/` 的 CSV/XLSX，自動配對 Input/Output，啟動 System Identification GUI。無檔案輸出。
- **`Create_Delta.m`** — 不確定性建模。讀 `2025.9.17 velocityIO_data/` 的 9 組速度資料，與名義模型比較計算相對誤差，用 ultidyn 隨機取樣產生 30 組不確定性模型。**輸出 `Delta_Data.mat`**（`z_all`, `p_all`, `k_all`, `Ts`）+ 三張比較圖。此檔需複製到 `Main/` 供 Python 使用。
- **`Plot_OE_Model.m`** — OE 模型精度評估。讀 `2025.9.17` 實測資料 + `OE Model/*.mat`，逐模型算 RMSE 並排名。**輸出** `OE Model/OE*.png`（各模型對比圖）、`OE_Model_RMSE_Result.csv`（排名表）、`OE_Model_RMSE_Ranking.png`（柱狀圖）。
- **`Check model/CheckModel.m`** — 讀 `Model_Data.m`，列印極點/零點、-3dB 頻寬、增益/相位裕度，畫波德圖。
- **`Check model/Cloop_sim.m`** — 讀 `Model_Data.m` + `CLoopsim.slx`，定義示意控制器，跑閉迴路 Simulink 模擬，畫誤差時域波形。

### 資料夾

- **`2025.9.15 ID data/`** — 早期鑑別資料（CSV/XLSX，1kHz 取樣）。供 `ID_Model.m`。`速度路徑資料.txt` 記載路徑配置。
- **`2025.9.17 velocityIO_data/`** — 主驗證資料（`Input/Output-velocity_1~9.csv`，1kHz）。核心資料集，供 `Create_Delta.m` 與 `Plot_OE_Model.m`。
- **`OE Model/`** — OE 多項式模型（`OE221.mat` 等 8 個）與 `Plot_OE_Model.m` 的評估產物。當前最佳模型為 OE222。
- **`Delta_Data.mat`** — 30 組不確定性模型的 zpk 資訊。由 `Create_Delta.m` 產生，被 `CNC.py` 的 `BUE_Plant()` 消費。
- **`Check model/CLoopsim.slx`** — Simulink 離散閉迴路模擬框架，由 `Cloop_sim.m` 調用。

---

## 資料檔案格式

| 檔案 | 產生者 | 內容 |
|------|--------|------|
| `Model/*.pth` | `Training.py` | PyTorch 字典 `{'iteration:N': {'actor', 'critic', 'FC', 'reward', …}}` |
| `ExperimentData/{時間戳}/simulation_data.npz` | `Simulation.py` | 每步 CC、FC、誤差、狀態、共振、模型切換點等 |
| `ExperimentData/{時間戳}/runtime_data.npz` | `Runtime.py` | 同上，上機全過程 |
| `Delta_Data.mat` | `Matlab/Create_Delta.m` | `z_all`, `p_all`, `k_all`, `Ts` |

**時間戳資料夾命名**：`{年}.{月}.{日}.{時}.{分}/`

### 雙模型機制

存在兩種訓練出的模型（檔名慣例）：
- `ModelBUE*.pth` — 在 BUE 不確定性上訓練，**追跡效果好**
- `ModelPRE*.pth` — 加隨機共振訓練，**抗共振能力強**

`Runtime.py` / `Simulation.py` 的 `use_switch_model` 開關：`True` 時偵測到共振就從追跡模型切到共振模型。

---

## 約定與注意事項

- **不要更動 State 維度 (131) 或 numFC (14)**，否則與已訓練的 `.pth` 不相容。`pdl=300`、`Ts=0.001` 是全域固定值。
- `Training.py`、`Runtime.py`、`Simulation.py` 各自有獨立的 `CNC_parameter`（reward 權重 `w_sumError` 等）與 `fft_limit_freq`，這些刻意不同，改一個不要連動改其他。
- `Model/` 與 `ExperimentData/` 已 gitignore（權重太大另存雲端，實驗資料量大）。
- 受控體模型參數源自 `Matlab/`。修改 plant 行為通常要回到 MATLAB 重新產生資料。
- **繪圖標準**（`Toolbox.py` / `Plot_Exp_Data.py` 一致）：圖框尺寸 `FIG_SIZE_SINGLE=(7.68, 5.76)`、`FIG_SIZE_WIDE=(11.52, 5.76)`、`FIG_SIZE_MULTI=(10.24, 10.24)`（需被 16 整除以相容影片編碼）；字體經 `matplotlib.rcParams` 統一（標題 24、軸標籤 20、刻度 18、圖例 18）。
- 改動行為時，請在 `版本紀錄.md` 增補對應條目。
