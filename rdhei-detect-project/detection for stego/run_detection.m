clear; clc;

%% ====== 路径设置（按需修改）======
in.stego = 'stego.png';                % 载密图
in.cover = 'cover.png';                % 可选：原图，仅用于对比可视化
in.mask  = 'location_map.png';         % 可选：嵌入位置真值（binary），用于评估
outdir   = 'results';
if ~exist(outdir,'dir'), mkdir(outdir); end

%% ====== 读取图像 ======
assert(exist(in.stego,'file')==2, '找不到 stego 图像：%s', in.stego);
I_stego = imread(in.stego);
I_stego = ensure_gray_double(I_stego);   % 转灰度 double ∈ [0,1]

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
% 残差（Laplacian + Sobel幅值），局部方差与局部熵（若无 entropyfilt 则退化为方差）
res_lap   = abs(imfilter(I_stego, fspecial('laplacian',0.2), 'replicate', 'same'));
Gx        = imfilter(I_stego, fspecial('sobel')','replicate');
Gy        = imfilter(I_stego, fspecial('sobel') ,'replicate');
res_sobel = sqrt(Gx.^2 + Gy.^2);

local_var = stdfilt(I_stego, true(5));           % 5×5 局部标准差（方差 proxy）
try
    local_ent = entropyfilt(im2uint8(I_stego), true(9));  % 9×9 局部熵（需要 IPT）
catch
    local_ent = local_var; % 无 entropyfilt 时退化
end

%% ====== 构建怀疑度热力图（可调权重）======
w = struct('lap', 0.6, 'sobel', 0.1, 'var', 0.5, 'ent', 0.3);  % 你可调参
S = w.lap*minmax_norm(res_lap) + ...
    w.sobel*minmax_norm(res_sobel) + ...
    w.var*minmax_norm(local_var) + ...
    w.ent*minmax_norm(local_ent);
S = minmax_norm(S);  % 归一化到 [0,1]

%% ====== 可视化与保存 ======
% 热力图
f1 = figure('Visible','off'); imagesc(S); axis image off; colormap hot; colorbar;
title('Suspicion Heatmap'); set(f1,'PaperPositionMode','auto');
saveas(f1, fullfile(outdir,'heatmap.png')); close(f1);

% 叠加图（热力伪彩覆盖到灰度图）
base_for_overlay = I_stego;
if ~isempty(I_cover), base_for_overlay = I_cover; end
overlay_img = overlay_heatmap(base_for_overlay, S, 0.55); % α 可调
imwrite(overlay_img, fullfile(outdir,'overlay.png'));

% 二值检测图（Otsu 阈值；你也可用分位数阈值）
th_otsu = graythresh(S);
BW = S >= th_otsu; 
imwrite(BW, fullfile(outdir,'binary_map.png'));

%% ====== （可选）评估：有真值掩膜时计算指标与ROC ======
if ~isempty(mask)
    % 基本分类指标（使用 Otsu 二值图）
    [prec, rec, f1] = prf1(BW, mask);
    % ROC & AUC（自实现，无需 perfcurve）
    [FPR, TPR, TH, AUC] = roc_curve(S, mask);

    % 保存 ROC 曲线
    f2 = figure('Visible','off'); plot(FPR,TPR,'LineWidth',2); grid on;
    xlabel('FPR'); ylabel('TPR'); title(sprintf('ROC (AUC = %.3f)', AUC));
    xlim([0 1]); ylim([0 1]);
    saveas(f2, fullfile(outdir,'roc.png')); close(f2);

    % 保存 CSV 指标
    T = table(prec, rec, f1, AUC, th_otsu, 'VariableNames', ...
        {'precision','recall','f1','AUC','otsu_threshold'});
    writetable(T, fullfile(outdir,'metrics.csv'));

    rocT = table(FPR, TPR, TH, 'VariableNames', {'FPR','TPR','threshold'});
    writetable(rocT, fullfile(outdir,'roc_points.csv'));
end

% 结束提示
fprintf('完成。结果已保存到：%s\n', outdir);