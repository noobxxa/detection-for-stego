clear; clc;

%% ====== ·�����ã������޸ģ�======
in.stego = 'stego.png';                % ����ͼ
in.cover = 'cover.png';                % ��ѡ��ԭͼ�������ڶԱȿ��ӻ�
in.mask  = 'location_map.png';         % ��ѡ��Ƕ��λ����ֵ��binary������������
outdir   = 'results';
if ~exist(outdir,'dir'), mkdir(outdir); end

%% ====== ��ȡͼ�� ======
assert(exist(in.stego,'file')==2, '�Ҳ��� stego ͼ��%s', in.stego);
I_stego = imread(in.stego);
I_stego = ensure_gray_double(I_stego);   % ת�Ҷ� double �� [0,1]

I_cover = [];
if exist(in.cover,'file')==2
    I_cover = imread(in.cover);
    I_cover = ensure_gray_double(I_cover);
    I_cover = imresize(I_cover, size(I_stego));
end

mask = [];
if exist(in.mask,'file')==2
    mask = imread(in.mask);
    mask = logical(imresize(mask>0, size(I_stego)));
end

%% ====== �в���ֲ�ͳ������ ======
% �вLaplacian + Sobel��ֵ�����ֲ�������ֲ��أ����� entropyfilt ���˻�Ϊ���
res_lap   = abs(imfilter(I_stego, fspecial('laplacian',0.2), 'replicate', 'same'));
Gx        = imfilter(I_stego, fspecial('sobel')','replicate');
Gy        = imfilter(I_stego, fspecial('sobel') ,'replicate');
res_sobel = sqrt(Gx.^2 + Gy.^2);

local_var = stdfilt(I_stego, true(5));           % 5��5 �ֲ���׼����� proxy��
try
    local_ent = entropyfilt(im2uint8(I_stego), true(9));  % 9��9 �ֲ��أ���Ҫ IPT��
catch
    local_ent = local_var; % �� entropyfilt ʱ�˻�
end

%% ====== �������ɶ�����ͼ���ɵ�Ȩ�أ�======
w = struct('lap', 0.6, 'sobel', 0.1, 'var', 0.5, 'ent', 0.3);  % ��ɵ���
S = w.lap*minmax_norm(res_lap) + ...
    w.sobel*minmax_norm(res_sobel) + ...
    w.var*minmax_norm(local_var) + ...
    w.ent*minmax_norm(local_ent);
S = minmax_norm(S);  % ��һ���� [0,1]

%% ====== ���ӻ��뱣�� ======
% ����ͼ
f1 = figure('Visible','off'); imagesc(S); axis image off; colormap hot; colorbar;
title('Suspicion Heatmap'); set(f1,'PaperPositionMode','auto');
saveas(f1, fullfile(outdir,'heatmap.png')); close(f1);

% ����ͼ������α�ʸ��ǵ��Ҷ�ͼ��
base_for_overlay = I_stego;
if ~isempty(I_cover), base_for_overlay = I_cover; end
overlay_img = overlay_heatmap(base_for_overlay, S, 0.55); % �� �ɵ�
imwrite(overlay_img, fullfile(outdir,'overlay.png'));

% ��ֵ���ͼ��Otsu ��ֵ����Ҳ���÷�λ����ֵ��
th_otsu = graythresh(S);
BW = S >= th_otsu; 
imwrite(BW, fullfile(outdir,'binary_map.png'));

%% ====== ����ѡ������������ֵ��Ĥʱ����ָ����ROC ======
if ~isempty(mask)
    % ��������ָ�꣨ʹ�� Otsu ��ֵͼ��
    [prec, rec, f1] = prf1(BW, mask);
    % ROC & AUC����ʵ�֣����� perfcurve��
    [FPR, TPR, TH, AUC] = roc_curve(S, mask);

    % ���� ROC ����
    f2 = figure('Visible','off'); plot(FPR,TPR,'LineWidth',2); grid on;
    xlabel('FPR'); ylabel('TPR'); title(sprintf('ROC (AUC = %.3f)', AUC));
    xlim([0 1]); ylim([0 1]);
    saveas(f2, fullfile(outdir,'roc.png')); close(f2);

    % ���� CSV ָ��
    T = table(prec, rec, f1, AUC, th_otsu, 'VariableNames', ...
        {'precision','recall','f1','AUC','otsu_threshold'});
    writetable(T, fullfile(outdir,'metrics.csv'));

    rocT = table(FPR, TPR, TH, 'VariableNames', {'FPR','TPR','threshold'});
    writetable(rocT, fullfile(outdir,'roc_points.csv'));
end

% ������ʾ
fprintf('��ɡ�����ѱ��浽��%s\n', outdir);