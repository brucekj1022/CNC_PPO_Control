%% ==== 一鍵模型驗證（極簡） ====
clear; clc

% 路徑與取樣時間
dataFolder = 'C:\Users\USER\Desktop\2026.4.24主線\Matlab\2025.9.15 ID data';
workFolder = 'C:\Users\USER\Desktop\2026.4.24主線\Matlab\';
cd(workFolder);
Ts = 1/1000;

files = [dir(fullfile(dataFolder,'*.xlsx')); dir(fullfile(dataFolder,'*.csv'))];

pairs = struct;   % pairs.(safeKey).input / .output / .dispName
for k = 1:numel(files)
    [~, base, ~] = fileparts(files(k).name);
    M  = readmatrix(fullfile(files(k).folder, files(k).name));
    if isempty(M), continue; end
    D  = M(:,2:end);
    D  = D(~all(isnan(D),2),:);

    % 例：20250915_16-26_Input-velocity 或 20250915_16-26_Output-velocity
    tok = regexpi(base, ...
        '^(?<id>\d{8}[_-]\d{2}[_-]\d{2})_(?<type>input|output)[_-]velocity$', ...
        'names');
    if isempty(tok), warning('跳過無法識別的檔名：%s', base); continue; end

    idDisp  = tok.id;                          % 原樣顯示
    safeKey = matlab.lang.makeValidName(['k_' idDisp]);  % 轉成合法欄位名
    type    = lower(tok.type);

    if ~isfield(pairs, safeKey)
        pairs.(safeKey) = struct('dispName', idDisp);
    end
    pairs.(safeKey).(type) = D(:,1);  % 取第1欄作為 u 或 y
end

% 建立 iddata
keys = fieldnames(pairs);
Dset = {}; names = {};
for i = 1:numel(keys)
    k = keys{i};
    if isfield(pairs.(k),'input') && isfield(pairs.(k),'output')
        u = pairs.(k).input;  y = pairs.(k).output;
        n = min(length(u),length(y));
        name = pairs.(k).dispName;
        Dset{end+1,1} = iddata(y(1:n), u(1:n), Ts, 'Name', name); %#ok<SAGROW>
        names{end+1,1} = name; %#ok<SAGROW>
    end
end

% Dset: 你的 iddata 清單 (cell)
maxN = 10;
n = min(maxN, numel(Dset));   % 實際要建立的數量

for i = 1:n
    eval(sprintf('d%d = Dset{%d};', i, i));
end

% 兩個離散模型（z^-1 形式）→ idtf
num1 = [0 0.03223 0.06099 -0.189 0.09699];
den1 = [1 -3.085 3.535 -1.782 0.3335];
sys1 = idtf(num1, den1, Ts); sys1.Name = '致維';

num2 = [0 0.1719 -0.3949 0.3047 -0.07076];
den2 = [1 -3.3051 4.1429 -2.3355 0.4995];
sys2 = idtf(num2, den2, Ts); sys2.Name = '紹平';

% 開啟 System Identification GUI（可直接 Import data/models）
ident
