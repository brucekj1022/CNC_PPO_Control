# CNC 使用 PPO — 進給軸控制器線上自動設計系統

CNC 進給軸控制器的線上自動設計系統。核心是用 **PPO 強化學習** 訓練一個 Actor，輸出「頻域增益限制點」(FC)，再透過 **QCQP 最佳化**（Youla/Q 參數化 + 互質分解 + 線性分式轉換 LFT，以 Gurobi 求解）即時合成控制器 `CC`，套用到 CNC 馬達受控體上，目標是壓制機台高頻共振同時維持追跡精度。

> 本專案註解、文件、版本紀錄皆為繁體中文。

---

## 目錄

- [控制流程](#控制流程)
- [專案目錄結構](#專案目錄結構)
- [執行方式](#執行方式)
  - [相依套件](#相依套件)
- [Main/ 各檔案說明](#main-各檔案說明)
- [Matlab/ 各檔案說明](#matlab-各檔案說明)
- [LabVIEW/ 各檔案說明](#labview-各檔案說明)
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
研究資料/
├── CLAUDE.md             # 給 Claude Code 的精簡指引（細節指向本檔）
├── README.md             # 本文件
│
├── Main/                 # Python 主程式（所有腳本都在此目錄下執行）
│   ├── CNC.py            # 系統核心：受控體模型、QCQP 控制器設計、繪圖輸出
│   ├── PPO_brain.py      # PPO 演算法：Actor/Critic 網路、ReplayBuffer
│   ├── Training.py       # 離線 PPO 訓練
│   ├── Simulation.py     # 離線閉迴路模擬
│   ├── Runtime.py        # 上機：TCP server 等 LabVIEW/cRIO 連線
│   ├── pc_server.py      # 通訊協議模組 + 獨立 TCP 連線測試工具
│   ├── Toolbox.py        # 工具選單（波德圖、極點、共振、路徑、Return 曲線）
│   ├── Plot_Exp_Data.py  # 視覺化實驗 npz（單一/批次統計）
│   ├── Test.py           # 測試用暫存檔（目前為空）
│   └── Delta_Data.mat    # 不確定性模型集合（由 Matlab 產生，CNC.py 載入）
│
├── Matlab/               # 系統鑑別、模型產生（受控體模型參數來源）
│   ├── Model_Data.m      # 名義模型定義（共用配置，硬編碼 ID 係數）
│   ├── ID_Model.m        # 系統辨識輔助工具（啟動 System ID GUI）
│   ├── Create_Delta.m    # 計算不確定性，產生 Delta_Data.mat
│   ├── Plot_OE_Model.m   # OE 模型精度評估與排名
│   ├── Delta_Data.mat    # 不確定性模型（需複製到 Main/ 供 Python 使用）
│   ├── 2025.9.15 ID data/         # 早期鑑別資料（CSV）
│   ├── 2025.9.17 velocityIO_data/ # 主驗證資料（9 組 Input/Output 速度）
│   ├── OE Model/         # OE 多項式模型（*.mat）與評估結果（*.png/csv）
│   └── Check model/      # 模型驗證（CheckModel.m / Cloop_sim.m / CLoopsim.slx）
│
├── LabVIEW/              # 上機端：cRIO 即時控制器（EtherCAT 連士林伺服）
│   ├── KJ_project.lvproj # LabVIEW 專案檔（主機 + cRIO 兩個 target）
│   ├── KJ_project.aliases# 連線 IP 別名（cRIO / 主機）
│   ├── Runtime.vi        # 上機主程式：TCP client 連 Main/Runtime.py
│   ├── Simulation.vi     # LabVIEW 端離線閉迴路模擬
│   ├── ID_X_velocityMode.vi # X 軸速度模式系統鑑別
│   ├── ID_Z_velocityMode.vi # Z 軸速度模式系統鑑別
│   ├── Save to csv_velocityMode.vi # 速度模式量測 I/O 存成 CSV
│   ├── pc_server.py      # 通訊協議模組副本（同 Main/pc_server.py）
│   ├── run_server.bat    # 啟動 pc_server.py 的批次檔
│   ├── 6Hz.txt / 8Hz.txt # 參考訊號資料（1 kHz × 30 秒）
│   ├── SDP-E_rev109(改).xml # EtherCAT 裝置描述檔（士林伺服 ESI）
│   └── test/TCP_test.vi  # TCP 連線測試
│
├── Model/                # 訓練權重 .pth（已 gitignore，太大另存雲端）
└── ExperimentData/       # 實驗資料（已 gitignore，量大）
```

---

## 執行方式

無 build / lint / test 框架。直接用 Python 執行各腳本。

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

## Main/ 各檔案說明

> 以下著重「這個檔案做什麼、會產出什麼、怎麼使用」。內部函數與參數細節請直接看原始碼註解。

### `CNC.py` — 系統核心

**功能**：整個系統的核心邏輯，定義受控體模型、路徑生成、QCQP 控制器設計、實驗繪圖。其他腳本都 import 它。
**會被 import，不單獨執行。** 主要對外提供四大類別：

- `CNCModel` — 受控體模型（ID / Test / BUE / PRE 四種，可選 X / Z 軸）
- `PathModel` — 參考路徑生成（chirp、訓練混合路徑等）
- `Costfunction` — QCQP 控制器設計（PPO 每步呼叫，解出控制器 `CC`）
- `PlotExporter` — 實驗繪圖輸出器

**產出**（透過 `PlotExporter`，存於 `../ExperimentData/<時間戳>/`）：

| 檔案 | 內容 |
|------|------|
| `frames/frame_NNN.png` | 每步 Bode 圖 + FC 點（動畫中間檔） |
| `frequency_response.mp4` | Bode 圖動畫 |
| `error.png` | 誤差波形圖 |
| `experiment_info.txt` | 實驗資訊文字 |

**使用方式**：不直接執行，由 `Training.py` / `Simulation.py` / `Runtime.py` / `Toolbox.py` 引用。

### `PPO_brain.py` — PPO 演算法

**功能**：定義 `PPO` 類別與 `ActorNet` / `CriticNet`（連續動作，Actor 輸出 mu/sigma）、`ReplayBuffer`。封裝動作取樣、優勢計算、Actor/Critic 更新與學習率調整。
**會被 import，不單獨執行。**
**產出**：無檔案輸出（模型權重由 `Training.py` 負責存檔）。
**使用方式**：被各執行腳本建立 `PPO(...)` 實例後呼叫。

### `Training.py` — 離線 PPO 訓練

**功能**：離線訓練 PPO Actor。用 `training_path`（20 條混合路徑）訓練，依 `lr_schedule` 分階段降學習率：**高學習率階段用 `ID_Plant`，低學習率階段切到 `PRE_Plant`（隨機共振）**。
**輸入**：`../Model/{read_file_name}`（預設 `ModelBUE1.pth`，可中斷續訓）、`Delta_Data.mat`。
**產出**：`../Model/{save_file_name}`（預設 `Model.pth`），每 100 輪追加一個 `iteration:N` 鍵；訓練結束顯示 Return 曲線。`enable_plot=True` 時另存 Bode 動畫與誤差圖。
**使用方式**：`python Training.py`，啟動後依提示輸入要從第幾輪續訓（直接 Enter 用最大輪）。

### `Simulation.py` — 離線閉迴路模擬

**功能**：載入已訓練模型，用單一 `test_path`（0~1 Hz chirp）跑完整離線閉迴路模擬，不需上機。`use_switch_model=True` 時偵測到共振會從追跡模型切到共振模型。
**輸入**：`../Model/{read_file_name}`（預設 `ModelPRE1.pth`）；雙模型模式讀兩個檔。
**產出**：`../ExperimentData/<時間戳>/simulation_data.npz`（每步 CC、FC、誤差、狀態、共振、模型切換點）+ `frequency_response.mp4` + `error.png`。
**使用方式**：`python Simulation.py`，依提示輸入起始輪數。

### `Runtime.py` — 上機實時運行

**功能**：與 `Simulation.py` 結構相同，但作為 **TCP server**（預設 `0.0.0.0:5005`）等 LabVIEW/cRIO 連線。每步收逗號分隔的誤差 `ek`（長度 = `pdl=300`）→ 合成新 `CC` → 傳回控制器係數。10 秒無連線會自動存檔並結束。
**輸入**：`../Model/{read_file_name}`（預設 `ModelBUE1.pth`）；上機過程的即時誤差由 TCP 傳入。
**產出**：`../ExperimentData/<時間戳>/runtime_data.npz`（上機全過程）+ `error.png`。
**使用方式**：`python Runtime.py`，依提示輸入起始輪數，等待 LabVIEW/cRIO client 連入。

### `pc_server.py` — 通訊協議模組 + 測試工具

**功能**：兩用途。(1) 提供 `array_to_str` / `str_to_array` / `recv` / `send` / `create_server` 等通訊函數，供 `Runtime.py` import。(2) 單獨執行時當獨立 TCP server，對每個連線固定回傳一組控制器係數，用於測試連線與量測 RTT。
**產出**：無檔案 I/O；單獨執行時於終端機印出收到的訊息與往返時間（RTT）。
**使用方式**：被 `Runtime.py` import；或 `python pc_server.py` 獨立測試連線（Ctrl+C 或輸入 `q` 結束）。

### `Toolbox.py` — 工具選單

**功能**：執行後顯示互動選單，集合各種繪圖與分析工具。
**輸入**：tool 5/6/7 需選擇實驗資料夾內的 `.npz`；tool 8 讀 `../Model/*.pth`。
**產出**：所有圖預設在螢幕顯示；tool 2/4/5/6/7 另存檔到所選實驗資料夾。

| 編號 | 功能 | 產出檔 |
|------|------|------|
| 1 | 受控體波德圖（ID/Test/BUE/PRE，可選 v2p/v2v） | 螢幕顯示 |
| 2 | 路徑資料繪圖與匯出 | `*.txt` / `*.xlsx` |
| 3 | 隨機共振峰值上下界繪圖（可選 v2p/v2v） | 螢幕顯示 |
| 4 | 動態 FFT 遮罩測試（產生動畫） | `animation.mp4` |
| 5 | 開迴路波德圖（可疊中央控制器） | `openloop_bode.png` |
| 6 | 閉迴路極點圖（含最小阻尼比線） | `closed_loop_poles.png` |
| 7 | 機台共振頻譜分析（互動式時間/濾波/異常值參數） | `error_full_fft.png` |
| 8 | 訓練 Return 曲線（可選全部或單一模型） | 螢幕顯示 |

**使用方式**：`python Toolbox.py`，輸入編號選功能，`q` 離開。

### `Plot_Exp_Data.py` — 實驗資料視覺化

**功能**：把 `Simulation.py` / `Runtime.py` 產生的 `.npz` 畫成圖。支援單一實驗（詳細圖）或多實驗（統計圖）。
**輸入**：`BATCH_MODE=True` 時批次處理檔案內 `BATCH_EXPERIMENTS` 清單；`False` 時彈窗手動選 `.npz` 檔或資料夾。
**產出**：

單實驗（`single`）模式，存到該實驗資料夾：

| 檔案 | 內容 |
|------|------|
| `experiment_info.txt` | 實驗基本資訊 + 每步詳細狀況表 |
| `reference_path.png` | 參考路徑時序圖 |
| `error.png` | 誤差時序圖（含事件標記線） |
| `controller_margins.png` | 4 合 1 性能指標：GM / PM / Wgc / 斜率 |
| `frequency_response.mp4` | 每步 Bode 圖 + FC 點動畫 |
| `error_fft.mp4` | 每步誤差 FFT 動畫 |

多實驗（`multi`）統計模式，存到指定資料夾：

| 檔案 | 內容 |
|------|------|
| `statistics_summary.txt` | 統計摘要（實驗數、RMS 統計、各次 RMS） |
| `error_statistics.png` | 誤差均值 + 標準差陰影 |
| `margins_statistics.png` | 4 合 1 性能指標均值 ±1σ |

**使用方式**：先在檔案頂端設定 `BATCH_MODE` 與 `BATCH_EXPERIMENTS`，再 `python Plot_Exp_Data.py`。

---

## Matlab/ 各檔案說明

受控體模型參數源自此資料夾。修改 plant 行為通常要回到 MATLAB 重新產生資料。

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

- **`Model_Data.m`** — 名義模型定義。依機台型號建立 `plantX`/`plantZ`（含 v2v、v2p、Ts），硬編碼 ID 係數。**無檔案輸出**，供其他腳本引用。
- **`ID_Model.m`** — 系統辨識輔助。掃描 `2025.9.15 ID data/` 的 CSV/XLSX，自動配對 Input/Output，啟動 System Identification GUI。**無檔案輸出。**
- **`Create_Delta.m`** — 不確定性建模。讀 `2025.9.17 velocityIO_data/` 的 9 組速度資料，與名義模型比較計算相對誤差，用 ultidyn 隨機取樣產生 30 組不確定性模型。**產出 `Delta_Data.mat`**（`z_all`, `p_all`, `k_all`, `Ts`）+ 三張比較圖。此檔需複製到 `Main/` 供 Python 使用。
- **`Plot_OE_Model.m`** — OE 模型精度評估。讀 `2025.9.17` 實測資料 + `OE Model/*.mat`，逐模型算 RMSE 並排名。**產出** `OE Model/OE*.png`、`OE_Model_RMSE_Result.csv`、`OE_Model_RMSE_Ranking.png`。
- **`Check model/CheckModel.m`** — 讀 `Model_Data.m`，列印極點/零點、-3dB 頻寬、增益/相位裕度，畫波德圖。
- **`Check model/Cloop_sim.m`** — 讀 `Model_Data.m` + `CLoopsim.slx`，定義示意控制器，跑閉迴路 Simulink 模擬，畫誤差時域波形。

### 資料夾

- **`2025.9.15 ID data/`** — 早期鑑別資料（CSV/XLSX，1 kHz）。供 `ID_Model.m`。
- **`2025.9.17 velocityIO_data/`** — 主驗證資料（`Input/Output-velocity_1~9.csv`，1 kHz）。核心資料集，供 `Create_Delta.m` 與 `Plot_OE_Model.m`。
- **`OE Model/`** — OE 多項式模型（`OE221.mat` 等）與 `Plot_OE_Model.m` 的評估產物。當前最佳模型為 OE222。
- **`Check model/CLoopsim.slx`** — Simulink 離散閉迴路模擬框架，由 `Cloop_sim.m` 調用。

---

## LabVIEW/ 各檔案說明

上機端（硬體側）程式。cRIO 即時控制器經 **EtherCAT** 驅動士林電機伺服馬達；`Runtime.vi` 當 **TCP client** 連到 Python 的 `Main/Runtime.py`（server），每步把追跡誤差傳給 Python、收回新合成的控制器係數套用到馬達。

> `.vi` 為 LabVIEW 二進位格式，需用 LabVIEW（專案以 25.0 版建立）開啟，無法用文字編輯器檢視。

### 程式 (VI)

- **`Runtime.vi`** — 上機主程式。當 TCP client 連 `Main/Runtime.py`，逐區段上傳誤差 `ek`、接收控制器係數，於 cRIO 上即時運行。
  **執行順序**：須**先**執行 Python 端 `Main/Runtime.py` 開啟 TCP server 監聽，**再**啟動本 VI 連線開始運行。
  > ⚠️ 目前 `Runtime.py` 每步的連線等待只設 **10 秒**，超時就自動存檔結束；務必在 Python 端開始監聽後盡快啟動 LabVIEW Runtime。
- **`Simulation.vi`** — LabVIEW 端的離線閉迴路模擬，不需連線真實機台。
- **`ID_X_velocityMode.vi` / `ID_Z_velocityMode.vi`** — X / Z 軸速度模式系統鑑別：對馬達送激勵訊號並記錄速度輸出，作為 MATLAB 系統辨識的輸入資料來源。
- **`Save to csv_velocityMode.vi`** — 把速度模式量測到的 Input/Output 存成 CSV（即 `Matlab/2025.9.17 velocityIO_data/` 那類資料）。
- **`test/TCP_test.vi`** — TCP 連線測試，搭配 `pc_server.py` 驗證主機與 cRIO 的通訊與往返時間。

### 設定與資料

- **`KJ_project.lvproj`** — LabVIEW 專案檔。含兩個 target：**My Computer**（主機端，放各 VI）與 **cRIO-Demo**（RT CompactRIO 即時控制器，含 EtherCAT 主站、兩台伺服從站、C 系列模組 Mod3 類比輸出 / Mod4 數位 I/O / Mod5 類比輸入）。
- **`KJ_project.aliases`** — 連線 IP 別名：cRIO `192.168.1.100`、主機 `192.168.100.60`。
- **`SDP-E_rev109(改).xml`** — EtherCAT 裝置描述檔（ESI），供 EtherCAT 主站辨識士林電機伺服驅動器從站。
- **`pc_server.py`** — `Main/pc_server.py` 的副本（版本略舊，邏輯相同），供上機端就近執行。
- **`run_server.bat`** — 啟動 `pc_server.py` 的批次檔（內含上機電腦的 Python 路徑，移機需自行修改）。
- **`6Hz.txt` / `8Hz.txt`** — 參考訊號資料（各 30000 點，1 kHz × 30 秒），供 LabVIEW 載入當輸入路徑。

---

## 資料檔案格式

| 檔案 | 產生者 | 內容 |
|------|--------|------|
| `Model/*.pth` | `Training.py` | PyTorch 字典 `{'iteration:N': {'actor', 'critic', 'FC', 'reward'}}` |
| `ExperimentData/<時間戳>/simulation_data.npz` | `Simulation.py` | 每步 CC、FC、誤差、狀態、共振、模型切換點等 |
| `ExperimentData/<時間戳>/runtime_data.npz` | `Runtime.py` | 同上，上機全過程 |
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
