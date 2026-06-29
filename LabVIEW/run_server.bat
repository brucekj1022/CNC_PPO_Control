@echo off
REM ===== 1) 選擇你的 Python 解譯器 =====
REM 如果用系統 Python：
set PY="C:\Users\Lab816\AppData\Local\Programs\Python\Python39\python.exe"
REM 如果你有虛擬環境，改成： set PY="C:\path\to\venv\Scripts\python.exe"

REM ===== 2) 切到腳本所在目錄 =====
cd /d "C:\Users\Lab816\Desktop\KJ\LabVIEW"

REM ===== 3) 執行伺服器 =====
%PY% pc_server.py

REM 視窗保留（若想自動關閉就刪掉這行）
pause