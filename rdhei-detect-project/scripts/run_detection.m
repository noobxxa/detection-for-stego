%% run_detection.m
% 传统非学习方法的隐写检测与可视化（MATLAB 仅用图像处理）
% 输入：examples/stego.png（必需），examples/cover.png（可选），examples/location_map.png（可选，二值真值）
% 输出：results/ 下的热力图、叠加图、二值图、ROC曲线与metrics.csv

clear; clc;

%% ====== 路径设置 ======
indir    = fullfile('..','examples');
outdir   = fullfile('..','results');
if ~exist(outdir,'dir'), mkdir(outdir); end

in.stego = fullfile(indir, 'stego.png');
in.cover = fullfile(indir, 'cover.png');
in.mask  = fullfile(indir, 'location_map.png');

%% ====== 读取图像 ======
assert(exist(in.stego,'file')==2, '找不到 stego 图像：%s', in.stego);
I_stego = imread(in.stego);
I_stego = ensure_gray_double(I_stego);

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

%% ====== 残差与局部统计特征 ======
res_lap   = abs(imfilter(I_stego, fspecial('laplacian',0.2), 'replicate', 'same'));
Gx        = imfilter(I_stego, fspecial('sobel')','replicate');
Gy        = imfilter(I_stego, fspecial('sobel') ,'replicate');
res_sobel = sqrt(Gx.^2 + Gy.^2);

local_var = stdfilt(I_stego, true(5));
try
    local_ent = entropyfilt(im2uint8(I_stego), true(9));
catch
    local_ent = local_var;
end

%% ====== 构建怀疑度热力图（权重可调）======
w = struct('lap', 0.6, 'sobel', 0.1, 'var', 0.5, 'ent', 0.3);
S = w.lap*minmax_norm(res_lap) + ...
    w.sobel*minmax_norm(res_sobel) + ...
    w.var*minmax_norm(local_var) + ...
    w.ent*minmax_norm(local_ent);
S = minmax_norm(S);

%% ====== 可视化与保存 ======
% 热力图
f1 = figure('Visible','off'); imagesc(S); axis image off; colormap hot; colorbar;
title('Suspicion Heatmap'); set(f1,'PaperPositionMode','auto');
saveas(f1, fullfile(outdir,'heatmap.png')); close(f1);

% 叠加图
base_for_overlay = I_stego;
if ~isempty(I_cover), base_for_overlay = I_cover; end
overlay_img = overlay_heatmap(base_for_overlay, S, 0.55);
imwrite(overlay_img, fullfile(outdir,'overlay.png'));

% 二值检测图（Otsu）
th_otsu = graythresh(S);
BW = S >= th_otsu;
imwrite(BW, fullfile(outdir,'binary_map.png'));

%% ====== （可选）评估：有真值掩膜时计算指标与ROC ======
if ~isempty(mask)
    [prec, rec, f1score] = prf1(BW, mask);
    [FPR, TPR, TH, AUC] = roc_curve(S, mask);

    f2 = figure('Visible','off'); plot(FPR,TPR,'LineWidth',2); grid on;
    xlabel('FPR'); ylabel('TPR'); title(sprintf('ROC (AUC = %.3f)', AUC));
    xlim([0 1]); ylim([0 1]);
    saveas(f2, fullfile(outdir,'roc.png')); close(f2);

    T = table(prec, rec, f1score, AUC, th_otsu, 'VariableNames', ...
        {'precision','recall','f1','AUC','otsu_threshold'});
    writetable(T, fullfile(outdir,'metrics.csv'));

    rocT = table(FPR, TPR, TH, 'VariableNames', {'FPR','TPR','threshold'});
    writetable(rocT, fullfile(outdir,'roc_points.csv'));
end

fprintf('完成。结果已保存到：%s\n', outdir);

%% ====== 本脚本用到的函数 ======
function I = ensure_gray_double(I)
    if size(I,3)==3, I = rgb2gray(I); end
    if ~isa(I,'double'), I = im2double(I); end
    I = max(0, min(1, I));
end

function Y = minmax_norm(X)
    X = double(X);
    mn = min(X(:)); mx = max(X(:));
    if mx>mn
        Y = (X-mn)/(mx-mn);
    else
        Y = zeros(size(X));
    end
end

function out = overlay_heatmap(gray01, score01, alpha)
    if ~isa(gray01,'double'), gray01 = im2double(gray01); end
    gray01 = repmat(gray01, [1 1 3]);
    cmap = hot(256);
    idx  = uint16(round(minmax_norm(score01)*255))+1;
    color = ind2rgb(idx, cmap);
    out = (1-alpha)*gray01 + alpha*color;
    out = max(0, min(1, out));
end

function [prec, rec, f1] = prf1(BW, GT)
    BW = logical(BW); GT = logical(GT);
    tp = sum(BW(:) & GT(:));
    fp = sum(BW(:) & ~GT(:));
    fn = sum(~BW(:) & GT(:));
    prec = tp / (tp + fp + eps);
    rec  = tp / (tp + fn + eps);
    f1   = 2*prec*rec / (prec+rec+eps);
end

function [FPR, TPR, TH, AUC] = roc_curve(score01, GT)
    GT = logical(GT); score01 = double(score01);
    TH = linspace(1, 0, 256);
    P = sum(GT(:)); N = numel(GT)-P;
    TPR = zeros(numel(TH),1); FPR = zeros(numel(TH),1);
    for k = 1:numel(TH)
        BW = score01 >= TH(k);
        TP = sum(BW(:) & GT(:));
        FP = sum(BW(:) & ~GT(:));
        TPR(k) = TP / (P + eps);
        FPR(k) = FP / (N + eps);
    end
    [FPR, idx] = sort(FPR); TPR = TPR(idx); TH = TH(idx);
    AUC = trapz(FPR, TPR);
end