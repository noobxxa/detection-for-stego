function [prec, rec, f1] = prf1(BW, GT)
    BW = logical(BW); GT = logical(GT);
    tp = sum(BW(:) & GT(:));
    fp = sum(BW(:) & ~GT(:));
    fn = sum(~BW(:) & GT(:));
    prec = tp / (tp + fp + eps);
    rec  = tp / (tp + fn + eps);
    f1   = 2*prec*rec / (prec+rec+eps);
end