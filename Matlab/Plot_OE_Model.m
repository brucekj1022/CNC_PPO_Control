%% ========================================================================
%  程式名稱：
%  自動載入 OE Model 並繪製頻率增益圖
%
%  程式作用：
%  1. 匯入實際機台 Input / Output velocity 資料
%  2. 由實際資料估測頻率增益 G
%  3. 自動讀取 OE Model 資料夾中的所有 OE 模型
%  4. 逐一計算各模型的頻率響應增益
%  5. 將每個 OE 模型與實際資料增益畫在同一張圖上
%  6. 以模型名稱作為圖標題，並自動存成 PNG 到 OE Model 資料夾
%  7. 計算每個 OE Model 與實際機台增益之 RMSE，排序後輸出 CSV
% ========================================================================

clc; clear; close all;

%% =========================
% 參數設定
% ==========================
Fs = 1000;                  % <-- 改成你的取樣頻率
point = 10000;
num_in_data = 9;

dataFolder = '2025.9.17 velocityIO_data';
modelFolder = 'OE Model';

% RMSE 計算方式固定使用線性增益，對應論文公式中的 |P(e^jwTs)|
rmseMode = 'linear';

%% =========================
% 引入實際機台資料
% ==========================
indata = cell(1, num_in_data);
outdata = cell(1, num_in_data);

for i = 1:num_in_data
    filename1 = fullfile(dataFolder, sprintf('Input-velocity_%d.csv', i));
    filename2 = fullfile(dataFolder, sprintf('Output-velocity_%d.csv', i));

    T1 = readtable(filename1);
    T2 = readtable(filename2);

    indata{i}  = T1{:,2};   % 第二欄為速度
    outdata{i} = T2{:,2};   % 第二欄為速度
end

%% =========================
% 實際增益
% ==========================
G = cell(1, num_in_data);

for i = 1:num_in_data
    [U, omega] = periodogram(indata{i}, [], point*2-1, Fs);
    [Y, ~]     = periodogram(outdata{i}, [], point*2-1, Fs);

    g = sqrt(Y ./ U);
    windowSize = 300;       % 視情況調整
    G{i} = movmean(g, windowSize);
end

%% =========================
% 自動讀取 OE Model 資料夾中的所有模型
% ==========================
modelFiles = dir(fullfile(modelFolder, '*.mat'));

if isempty(modelFiles)
    error('在資料夾 "%s" 中找不到任何 .mat 模型檔', modelFolder);
end

fprintf('共找到 %d 個模型檔：\n', length(modelFiles));
for k = 1:length(modelFiles)
    fprintf('%d. %s\n', k, modelFiles(k).name);
end

%% =========================
% RMSE 結果儲存空間
% ==========================
rmseResults = table('Size', [0 5], ...
    'VariableTypes', {'string', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'ModelName', 'MeanRMSE', 'StdRMSE', 'MinRMSE', 'MaxRMSE'});

%% =========================
% 逐一讀取模型並畫圖、存圖
% ==========================
for k = 1:length(modelFiles)

    modelPath = fullfile(modelFolder, modelFiles(k).name);
    S = load(modelPath);

    % 取得 .mat 內所有變數名稱
    varNames = fieldnames(S);

    % 找出第一個可當作動態系統模型的變數
    currentModel = [];

    for v = 1:length(varNames)
        obj = S.(varNames{v});

        if isa(obj, 'idtf') || isa(obj, 'idpoly') || isa(obj, 'tf') || ...
           isa(obj, 'zpk')  || isa(obj, 'ss')     || isa(obj, 'idss')
            currentModel = obj;
            break;
        end
    end

    if isempty(currentModel)
        warning('檔案 "%s" 中找不到可用的模型物件，跳過。', modelFiles(k).name);
        continue;
    end

    % 若是識別模型，轉成 tf 比較保險
    try
        TF_current = tf(currentModel);
    catch
        warning('檔案 "%s" 的模型無法轉成 tf，跳過。', modelFiles(k).name);
        continue;
    end

    %% 計算模型增益
    [magModel, ~, ~] = bode(TF_current, omega*2*pi);
    magModel = squeeze(magModel);

    %% 計算此模型與實際機台增益的 RMSE
    magModel = magModel(:);
    rmseEachData = zeros(num_in_data, 1);

    for i = 1:num_in_data
        realGain = G{i}(:);

        % 避免 log10(0) 或除頻譜造成的 Inf / NaN
        validIdx = isfinite(realGain) & isfinite(magModel) & ...
                   realGain > 0 & magModel > 0;

        switch lower(rmseMode)
            case 'db'
                realCompare  = 20*log10(realGain(validIdx));
                modelCompare = 20*log10(magModel(validIdx));
            case 'linear'
                realCompare  = realGain(validIdx);
                modelCompare = magModel(validIdx);
            otherwise
                error('rmseMode 只能設定為 ''dB'' 或 ''linear''。');
        end

        rmseEachData(i) = sqrt(mean((realCompare - modelCompare).^2));
    end

    meanRMSE = mean(rmseEachData);
    stdRMSE  = std(rmseEachData);
    minRMSE  = min(rmseEachData);
    maxRMSE  = max(rmseEachData);

    %% 模型名稱
    modelTitle = erase(modelFiles(k).name, '.mat');

    rmseResults = [rmseResults; ...
        {string(modelTitle), meanRMSE, stdRMSE, minRMSE, maxRMSE}]; %#ok<AGROW>

    %% 畫圖
    fig = figure('Name', modelTitle, ...
                 'NumberTitle', 'off', ...
                 'Color', 'w', ...
                 'Position', [100, 100, 1000, 750]);

    legend_strings = cell(1, num_in_data + 1);

    for i = 1:num_in_data
        semilogx(omega, 20*log10(G{i}), 'LineWidth', 1.2);
        hold on;
        legend_strings{i} = sprintf('Data %d', i);
    end

    semilogx(omega, 20*log10(magModel), 'r', 'LineWidth', 2.5);
    legend_strings{end} = modelTitle;

    legend(legend_strings, 'Location', 'best', 'FontSize', 28);
    xlabel('Frequency (Hz)', 'FontSize', 36, 'FontWeight', 'bold');
    ylabel('Gain (dB)', 'FontSize', 36, 'FontWeight', 'bold');
    title(modelTitle, ...
          'FontSize', 44, 'FontWeight', 'bold', 'Interpreter', 'none');
    set(gca, 'FontSize', 32, 'LineWidth', 1.2);
    grid on;

    %% 存圖到 OE Model 資料夾
    pngFile = fullfile(modelFolder, [modelTitle '.png']);
    exportgraphics(fig, pngFile, 'Resolution', 300);

    %% 關閉圖窗
    close(fig);
end

%% =========================
% RMSE 排序、輸出與畫圖
% ==========================
rmseResults = sortrows(rmseResults, 'MeanRMSE', 'ascend');

fprintf('\n===== OE Model RMSE 排名，越小越接近實機 =====\n');
disp(rmseResults);

bestModel = rmseResults.ModelName(1);
bestRMSE  = rmseResults.MeanRMSE(1);
fprintf('最佳模型：%s，平均 RMSE = %.6g %s\n', bestModel, bestRMSE, rmseMode);

% 存成 CSV，方便貼到論文或 Excel 整理
rmseCsvFile = fullfile(modelFolder, 'OE_Model_RMSE_Result.csv');
writetable(rmseResults, rmseCsvFile);

% 畫 RMSE 排名圖
figRMSE = figure('Name', 'OE Model RMSE Ranking', ...
                 'NumberTitle', 'off', ...
                 'Color', 'w', ...
                 'Position', [100, 100, 1200, 650]);

bar(categorical(rmseResults.ModelName), rmseResults.MeanRMSE);
grid on;
xlabel('OE Model', 'FontSize', 28, 'FontWeight', 'bold');
ylabel(sprintf('Mean RMSE (%s)', rmseMode), 'FontSize', 28, 'FontWeight', 'bold');
title('OE Model RMSE Ranking', 'FontSize', 34, 'FontWeight', 'bold');
set(gca, 'FontSize', 20, 'LineWidth', 1.2);
xtickangle(45);

rmsePngFile = fullfile(modelFolder, 'OE_Model_RMSE_Ranking.png');
exportgraphics(figRMSE, rmsePngFile, 'Resolution', 300);
close(figRMSE);

fprintf('所有圖片已儲存到資料夾：%s\n', modelFolder);
fprintf('RMSE 結果已儲存：%s\n', rmseCsvFile);
fprintf('RMSE 排名圖已儲存：%s\n', rmsePngFile);