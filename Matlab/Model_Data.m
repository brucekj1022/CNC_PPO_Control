%% ========================================================================
%  程式名稱：
%  統一模型載入程式
%
%  程式作用：
%  1. 在程式開頭選擇要使用的模型來源
%  2. 根據選定的模型，自動建立 plantX / plantZ / Ts 等變數
%  3. 供其他程式直接引用
%
%  使用方式：
%  - 只需修改下方 model_name
%  - 執行後會輸出對應的 plantX / plantZ / Ts
%
%  可選模型：
%  - '致維'
%  - '紹平'
%  - '楷鈞'
%
%  備註：
%  - '楷鈞' 模型目前只有 X 軸，沒有 Z 軸
% ========================================================================

%% 選擇模型
model_name = '楷鈞';   % 可改成 '致維'、'紹平'、'楷鈞'

%% 基本參數
Ts = 0.001;
z = tf('z', Ts);
Vmax = 300;

integrator = tf([Ts, 0], [1, -1], z);
rpm2mms_z = 12 / 60;
rpm2mms_x = 10 / 60;

%% 依選擇載入模型
switch model_name

    case '致維'
        % transfer function (x axis) (rpm -> rpm)
        Pv_x = tf([0, 0.032229639522932, 0.060985247548587, -0.189017468410220, 0.096991370106707], ...
                  [1, -3.084824116823246, 3.534522236179820, -1.782062555742226, 0.333547393790890], z);

        % transfer function (z axis) (rpm -> rpm)
        Pv_z = tf([0, 0.0973217504156460, -0.209580502231482, 0.151454774135651, -0.0370139184925609], ...
                  [1, -3.44700230387150, 4.55138855100522, -2.73113485001183, 0.628907804339994], z);

        % transfer function (rpm -> mm/s)
        Px = Pv_x * rpm2mms_x * integrator;
        Pz = Pv_z * rpm2mms_z * integrator;

        plantX = struct('v2p', Px, 'v2v', Pv_x, 'Ts', Ts);
        plantZ = struct('v2p', Pz, 'v2v', Pv_z, 'Ts', Ts);

    case '紹平'
        % z axis (v -> v)
        Pv_z = tf([0, 0.165528814275144, -0.422246806441294, 0.359429000517177, -0.101574613814810], ...
                  [1, -3.548330884241820, 4.767233602436503, -2.874814273749141, 0.657067148225129], z);

        % x axis (v -> v)
        Pv_x = tf([0, 0.171925943013834, -0.394933146268344, 0.304708442767194, -0.0797623242872279], ...
                  [1, -3.30507886927150, 4.14294763501288, -2.33547736526031, 0.499524269949438], z);

        % transfer function (rpm -> mm/s)
        Px = Pv_x * rpm2mms_x * integrator;
        Pz = Pv_z * rpm2mms_z * integrator;

        plantX = struct('v2p', Px, 'v2v', Pv_x, 'Ts', Ts);
        plantZ = struct('v2p', Pz, 'v2v', Pv_z, 'Ts', Ts);

    case '楷鈞'
        % transfer function (x axis) (rpm -> rpm)
        Pv_x = tf([0, 0.0410789388950551, 0.116567016168533], ...
                  [1, -1.41353788543924, 0.566877911191563], z);   % oe221

        % transfer function (rpm -> mm/s)
        Px = Pv_x * rpm2mms_x * integrator;

        plantX = struct('v2p', Px, 'v2v', Pv_x, 'Ts', Ts);

        % 楷鈞版本目前沒有 Z 軸模型
        plantZ = [];

    otherwise
        error('未知的 model_name：%s。請使用 ''致維''、''紹平'' 或 ''楷鈞''。', model_name);
end

%% 清除中間變數
clear Px Pz Pv_x Pv_z integrator rpm2mms_x rpm2mms_z z;