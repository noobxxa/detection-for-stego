function [FPR, TPR, TH, AUC] = roc_curve(score01, GT)
    % 生成 ROC（不依赖 perfcurve）
    GT = logical(GT); score01 = double(score01);
    TH = linspace(1, 0, 256); % 从高到低阈值
    P = sum(GT(:)); N = numel(GT)-P;
    TPR = zeros(numel(TH),1); FPR = zeros(numel(TH),1);
    for k = 1:numel(TH)
        BW = score01 >= TH(k);
        TP = sum(BW(:) & GT(:));
        FP = sum(BW(:) & ~GT(:));
        TPR(k) = TP / (P + eps);
        FPR(k) = FP / (N + eps);
    end
    % AUC（按 FPR 升序积分）
    [FPR, idx] = sort(FPR); TPR = TPR(idx); TH = TH(idx);
    AUC = trapz(FPR, TPR);
end
