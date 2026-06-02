Model
plant=plantX.v2v;
poles=pole(plant);
zeros=zero(plant);
disp(poles);
disp(zeros);

% -3dB 頻寬
wb = bandwidth(plant);   % 單位 rad/s
wb_Hz = wb / (2*pi);   % 換算成 Hz
fprintf('-3 dB 頻寬 = %.2f Hz\n', wb_Hz);

% 增益裕度 / 相位裕度
[GM, PM, Wcg, Wcp] = margin(plant);
GM_dB = 20*log10(GM);    % dB
fprintf('增益裕度: %.2f dB (at %.1f Hz)\n', GM_dB, Wcg/(2*pi));
fprintf('相位裕度: %.1f deg (at %.1f Hz)\n', PM, Wcp/(2*pi));

figure;
bode(plant); grid on;

hold on;
[mag,~,w] = bode(plant);
mag = squeeze(mag);
[~,idx] = min(abs(20*log10(mag) - (-3)));
plot(w(idx), mag(idx), 'ro');   % -3 dB 點


