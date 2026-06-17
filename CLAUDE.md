# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> 本專案註解、文件、版本紀錄皆為繁體中文，回覆與新增註解請沿用繁中。

## 專案概述

CNC 進給軸控制器的線上自動設計系統。核心是用 **PPO 強化學習** 訓練一個 Actor，輸出「頻域增益限制點」(FC)，再透過 **QCQP 最佳化**（Youla/Q 參數化 + 互質分解 + 線性分式轉換 LFT，以 Gurobi 求解）即時合成控制器 `CC`，套用到 CNC 馬達受控體上，目標是壓制機台高頻共振同時維持追跡精度。

控制流程：`State → Actor → FC → QCQP(Costfunction) → 控制器 CC → 模擬/上機 → 誤差/共振 → 下一個 State`

- **State** (131 維): `[action(28), path_FFT(100), maxResonance(2), sumError(1)]`
- **Action** (28 維): `numFC=14` 個限制點 × (頻率, 增益)，Actor 輸出為 dB，外部轉成線性值

## 執行方式

無 build / lint / test 框架（這是 Visual Studio Python 專案 `CNC使用PPO.pyproj`，啟動檔為 `Training.py`）。直接用 Python 跑各腳本。

**重要：所有腳本都必須在 `Main/` 目錄下執行**，因為使用相對路徑：
- `Delta_Data.mat` 從工作目錄載入（`Main/Delta_Data.mat`）
- 模型權重讀寫 `../Model/`，實驗資料寫 `../ExperimentData/`

```bash
cd Main
python Training.py     # 離線 PPO 訓練（讀 ../Model/*.pth，每 100 輪存檔）
python Simulation.py   # 離線閉迴路模擬（用 PRE_Plant 等模型，輸出 simulation_data.npz）
python Runtime.py      # 上機：TCP server 等 LabVIEW/cRIO 連線，每步收 ek、回傳 CC 係數
python Toolbox.py      # 互動式分析選單（波德圖、極點、共振、路徑、產生測試控制器）
python pc_server.py    # 獨立 TCP 連線測試（固定回傳一組控制器係數）
python Plot_Exp_Data.py  # 視覺化 runtime_data.npz（單一或批次統計）
python Plot_return.py    # 畫訓練 reward 曲線
python Plot_Full_FFT.py  # 誤差頻譜分析
```

## 相依套件

無 `requirements.txt`。需要：`numpy`, `scipy`, `control`, `torch`, `gurobipy`, `imageio`, `matplotlib`, `pandas`, `openpyxl`。

- **`gurobipy` 需要有效的 Gurobi 授權** — 控制器合成的 QCQP 求解完全依賴它，沒授權無法跑 `Costfunction`。
- `torch` 自動偵測 CUDA，無 GPU 會 fallback 到 CPU。

## 核心架構：`Main/CNC.py`

整個系統的核心邏輯都在這個模組，其他腳本只是不同的執行入口（訓練/模擬/上機/繪圖）。

**模組層級函數**
- `SimulateResponse(path, CC, plant, X0, Ts)` — 模擬閉迴路響應，回傳 `(下一步狀態, 誤差um, 輸出)`
- `find_Wgc(CC, plant, Ts)` — 用切比雪夫多項式法求增益交越頻率
- `find_resonance(CC, plant, Ts, path_section, error)` — 從誤差頻譜辨識高頻共振點（頻率 > Wgc 且振幅 > 5×輸出均值）

**`CNCModel`** — 受控體模型（軸別 `'x'`/`'z'`，目前主用 X 軸）
- `ID_Plant()` — 系統鑑別得到的標稱模型（參數來自 MATLAB ID）
- `test_Plant()` — 標稱 + 固定共振點
- `BUE_Plant()` — 從 `Delta_Data.mat` 隨機抽一組不確定性模型（Base Uncertainty Ensemble）
- `PRE_Plant()` — BUE + 隨機高頻共振（Perturbed Resonant Ensemble）

**`PathModel`** — 路徑生成：`test_path`(0~1Hz chirp)、`test_path2`(0~8Hz)、`training_path`(20 條混合)、`up_down_chirp`

**`Costfunction`** — QCQP 控制器設計（最重要的類別）
- 建構時做互質分解 (`coprime_factorization_ss`) + LFT (`linear_fractional_transformation`)，並用中央控制器初始化
- `switch_controller(path, path_index, FC, ek)` — **PPO 每步的主要入口**：解 QCQP + `suppress_resonance` 共振壓制，回傳 `(status, CC, ek_hat, manual_add_FC)`
- `optimizationcvx()` — 實際的 Gurobi QCQP 求解（時域誤差最小化 + 頻域增益上下界約束）
- `reward()` — PPO 獎勵：Solved 看誤差與 FC 均勻度，Infeasible 看與上次可行 FC 的差距
- `resonanceTable` — 共振紀錄表，含「真共振確認步數」與「延遲倒數」狀態機（見版本紀錄 v2026.4.24）
- `status` 三態：`"Solved"` / `"semiSolved"`（加共振壓制後 QCQP 無解，沿用舊 CC）/ `"Infeasible"`

**`PlotExporter`** — 在 `../ExperimentData/<時間戳>/` 下產生 Bode 圖框架、MP4 動畫、誤差圖、實驗資訊

## PPO：`Main/PPO_brain.py`

`PPO` 類別 + `ActorNet`/`CriticNet`（連續動作，Actor 輸出 mu/sigma，`bound = 20*log10(3000)` dB）+ `ReplayBuffer`。`Training.py` 用 `lr_schedule` 分階段降學習率，**高學習率階段用 `ID_Plant` 訓練、低學習率階段切到 `PRE_Plant`（隨機共振）**。

## 雙模型機制

存在兩種訓練出的模型（檔名慣例）：
- `ModelBUE*.pth` — 在 BUE 不確定性上訓練，**追跡效果好**
- `ModelPRE*.pth` — 加隨機共振訓練，**抗共振能力強**

`Runtime.py` / `Simulation.py` 的 `use_switch_model` 開關：`True` 時偵測到共振就從追跡模型切到共振模型。

## 上機通訊：`Main/pc_server.py`

`Runtime.py` 當 TCP server（預設 `0.0.0.0:5005`），LabVIEW/cRIO 當 client。每步：收逗號分隔的誤差 `ek`（長度 = `pdl=300`）→ 算新 `CC` → 用 `array_to_str` 傳回控制器係數（先傳 4 位數長度標頭再傳資料）。

## 約定與注意事項

- **不要更動 State 維度 (131) 或 numFC (14)**，否則與已訓練的 `.pth` 不相容。`pdl=300`、`Ts=0.001` 是全域固定值。
- `Training.py`、`Runtime.py`、`Simulation.py` 各自有獨立的 `CNC_parameter`（reward 權重 `w_sumError` 等）與 `fft_limit_freq`，這些刻意不同，改一個不要連動改其他。
- `Model/` 與 `ExperimentData/` 已 gitignore（權重太大另存雲端，實驗資料量大）。
- 受控體模型參數源自 `Matlab/`（系統鑑別、OE 模型、產生 `Delta_Data.mat`）。修改 plant 行為通常要回到 MATLAB 重新產生資料。
- 改動行為時，請在 `版本紀錄.md` 增補對應條目（專案以此追蹤每版的模型/參數/流程差異）。
