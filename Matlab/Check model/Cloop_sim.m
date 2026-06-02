 clear;
%% ===== 一鍵閉迴路追蹤測試：C(z) 串 P(z) + 負回授 =====
Model
Ts   = 0.001;                 % 取樣時間
Tend = 15;                     % 模擬時間
f0 = 0; f1 = 1;               % chirp 0→1 Hz
t = (0:Ts:Tend)';             
r = 10 * chirp(t, f0, Tend, f1, 'linear', -90)+2;   % 參考路徑（振幅10、正弦起始）
r = timeseries(r,t); 
%% === 你的模型（以 z 的正次方表示；係數由高次→常數） ===
numC_z = [1 1959.302 -2865.4329 1892.9066 -1577.0496 1256.1066 6994.444 -10466.283 -3151.8682 10563.204 -4494.296 243.16862 -69.72598];
denC_z =  [1 -1.2839447 0.6892322 -0.10934689 -0.004581599 -0.013008632 -0.0014074598 -0.052522484 -0.19558156 -0.14390853 -0.0008565978 -0.0023896238 0];
numC_z = [ 3507.6113, -4875.3955, 2501.5308, -2374.749, 1693.5095, 302.21582, ...
         1003.8795, -2487.2388, 1251.1041, -920.4005, 1200.3229, -516.91455, ...
        -0.000036743924 ];

denC_z = [ 1., -1.3079528, 0.6099558, -0.15709905, -0.051492758, -0.016081883, ...
        -0.000000002938381, -0.010085748, -0.035933144, -0.026996037, ...
        -0.017715514, -0.000000001259176, -0. ];
% Plant P(z)
numP_z = [0,	0.0410789388950551,	0.116567016168533];
denP_z = [1,	-1.41353788543924,	0.566877911191563];
z = tf('z',Ts);
CC=tf(numC_z, denC_z, z);
assignCase = 2;
switch assignCase
    case 1
        Plant=plantX.v2p;
    case 2
        integrator = tf([Ts,0],[1,-1],z);
        Plant=tf(numP_z,denP_z,z)*(10/60)*integrator;
end
%% ===組合CLoop===
OLoop = tf(CC*Plant);                 % 開迴路 TF（確保是 tf 物件）
assignCase = 2;
switch assignCase
    case 1
        [num, den] = tfdata(OLoop, 'v');      % 取係數（向量）
        % 若分子次數較低，在「左側」補 0 對齊（等同 np.pad(num, (len(den)-len(num), 0))）
        if length(num) < length(den)
            num = [zeros(1, length(den) - length(num)), num];
        end
        % 負回授：T(z) = num / (den + num)
        CLoop = tf(num, den + num, Ts);

    case 2
        CLoop=minreal(OLoop/(1+OLoop));
end
poles=pole(CLoop);

%% ===開始模擬===
out = sim('CLoopsim', 'StartTime','0', 'StopTime','15');
y = out.y; 
e = out.e;

% 轉成向量並畫圖
t = y.Time;  yv = y.Data;  ev = e.Data;
%figure; plot(t,yv); grid on; title('y')
figure; plot(t,ev*1000); grid on; title('e = r - y')