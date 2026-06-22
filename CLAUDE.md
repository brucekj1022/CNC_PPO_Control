# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> 本專案註解與文件皆為繁體中文，回覆與新增註解請沿用繁中。

完整架構說明、目錄結構、各腳本 I/O、Matlab 流程請見 [README.md](README.md)。

## 執行環境

無 build / lint / test 框架。所有 Python 腳本都必須在 `Main/` 目錄下執行（相對路徑 `../Model/`、`../ExperimentData/`）。

## 改動程式碼時的注意事項

- **不要更動 State 維度 (131) 或 numFC (14)**，否則與已訓練的 `.pth` 不相容。`pdl=300`、`Ts=0.001` 是全域固定值。
- `Training.py`、`Runtime.py`、`Simulation.py` 各自有獨立的 `CNC_parameter`（`w_sumError` 等）與 `fft_limit_freq`，**這些刻意不同，改一個不要連動改其他**。
- 修改 plant 行為須回到 `Matlab/` 重新產生 `Delta_Data.mat`。

## CNC.py 補充細節（README 未涵蓋）

`Costfunction` 建構時會執行互質分解（`coprime_factorization_ss`）+ LFT（`linear_fractional_transformation`），並以中央控制器作為 Q 參數初值。

- `status` 三態：`"Solved"` / `"semiSolved"`（共振壓制後 QCQP 仍無解，沿用舊 CC）/ `"Infeasible"`
- `resonanceTable` — 共振紀錄表，含「真共振確認步數」與「延遲倒數」狀態機
- `CNCModel` 各方法回傳 `{'v2p': …, 'v2v': …, 'Ts': …}`（v2p=速度→位置，v2v=速度→速度）
