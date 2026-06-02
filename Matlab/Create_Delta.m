%% ========================================================================
%  程式名稱：
%  實際機台資料、ID Model 與不確定性模型分析
%
%  程式功用：
%  1. 匯入實際機台運行資料（Input / Output velocity）
%  2. 載入 ID Model，取得名義模型 TF_ID
%  3. 計算實際資料的頻率增益
%  4. 比較實際資料與 ID Model 的頻率響應
%  5. 根據 DeltaG 建立權重函數 w
%  6. 建立不確定性模型並觀察其頻率響應
%  7. 視需要將不確定性模型資料存出
% ========================================================================

clc;
clear;
close all;

%% =========================
%匯入實際機台運行資料和ModelData
% ==========================
Model_Data;
TF_ID = plantX.v2v;
Ts = plantX.Ts;
Fs = 1 / Ts;

num_in_data = 9;   % 輸入資料幾筆
indata = cell(1, num_in_data);
outdata = cell(1, num_in_data);

for i = 1:num_in_data
    filename1 = sprintf('2025.9.17 velocityIO_data/Input-velocity_%d.csv', i);
    filename2 = sprintf('2025.9.17 velocityIO_data/Output-velocity_%d.csv', i);

    T = readtable(filename1);      % 讀入輸入資料表
    indata{i} = T{:, 2};           % 第二欄為速度

    T = readtable(filename2);      % 讀入輸出資料表
    outdata{i} = T{:, 2};          % 第二欄為速度
end

%% =========================
% 實際增益和 ID 增益
% ==========================
point = 10000;

% 實際增益
G = cell(1, num_in_data);

for i = 1:num_in_data
    [U, omega] = periodogram(indata{i}, [], point * 2 - 1, Fs);
    [Y, ~] = periodogram(outdata{i}, [], point * 2 - 1, Fs);

    g = sqrt(Y ./ U);

    windowSize = 300;              % 視情況調整（越大越平滑）
    G{i} = movmean(g, windowSize);
end

% ID 增益
[magID, PHASE, ~] = bode(TF_ID, omega * 2 * pi);
magID = squeeze(magID);

%% =========================
% 畫增益圖
% ==========================
figure;
legend_strings = cell(1, num_in_data);

for i = 1:num_in_data
    semilogx(omega, 20 * log10(G{i}));
    legend_strings{i} = sprintf('Data %d', i);
    hold on;
end

semilogx(omega, 20 * log10(magID), 'r', 'LineWidth', 2);
legend_strings{end + 1} = 'Nominal';   % 最後一條紅線

legend(legend_strings, 'Location', 'best');
xlabel('Frequency (Hz)');
ylabel('Gain(dB)');
title('Estimated Gain via Power Spectrum');
grid on;

%% =========================
% 依照最大的 DeltaG 建立權重 w
% ==========================
% DeltaG 的 magnitude
mag_DeltaG = zeros(num_in_data, point);

for i = 1:num_in_data
    mag_DeltaG(i, :) = abs((G{i} - magID) ./ magID);%這邊跟論文稍微不太一樣，這裡是純增益，論文上有帶相角，可以近似
end

% 依照最大的 DeltaG 建立權重 w
s = tf('s');
A = 0.02 * (s / 5 + 1) * (s / 200 + 1);
B = (s / 110 + 1) * (s / 2200 + 1);
w = A / B;
w = c2d(w, Ts, 'tustin');

[magw, ~] = bode(w, omega * 2 * pi);
magw = squeeze(magw);

%% =========================
% 畫 DeltaG 跟權重 w
% ==========================
figure;
legend_strings = cell(1, num_in_data);

for i = 1:num_in_data
    semilogx(omega, mag_DeltaG(i, :));
    legend_strings{i} = sprintf('DeltaP %d', i);
    hold on;
end

semilogx(omega, magw, 'r', 'LineWidth', 2);
legend_strings{end + 1} = 'Weight';    % 最後一條紅線

legend(legend_strings, 'Location', 'best');
xlabel('Frequency (Hz)');
ylabel('Gain');
title('DeltaP & Weight');
grid on;

%% =========================
% 建立不確定性模型
% ==========================
num_out_data = 30;   % 輸出多少的不確定性模型
model_unc = ultidyn('Delta', [1 1], 'Bound', 1);

mag_Unc = zeros(num_out_data, point);
delta = cell(1, num_out_data);

for i = 1:num_out_data
    unc = usample(model_unc);
    unc = c2d(unc, Ts, 'tustin');

    TF_uncertainty = TF_ID * (1 + w * unc);
    delta{i} = 1 + w * unc;   % 直接當成一個系統

    [mag, ~] = bode(TF_uncertainty, omega * 2 * pi);
    mag_Unc(i, :) = squeeze(mag);
end

%% =========================
% 畫不確定性模型增益圖
% ==========================
figure;
legend_strings = cell(1, num_in_data);

for i = 1:num_out_data
    semilogx(omega, 20 * log10(mag_Unc(i, :)));
    legend_strings{i} = sprintf('Unc %d', i);
    hold on;
end

semilogx(omega, 20 * log10(magID), 'r', 'LineWidth', 2);
legend_strings{end + 1} = 'TF_ID';   % 最後一條紅線

% legend(legend_strings, 'Location', 'best');
xlabel('Frequency (Hz)');
ylabel('Gain(dB)');
title('Uncertainty Plant');
grid on;

%% =========================
% 存出資料
% ==========================
%{
z_all = cell(num_out_data, 1);
p_all = cell(num_out_data, 1);
k_all = zeros(num_out_data, 1);

for i = 1:num_out_data
    [z_all{i}, p_all{i}, k_all(i)] = zpkdata(delta{i}, 'v');
end

save('delta.mat', 'z_all', 'p_all', 'k_all', 'Ts');
%}