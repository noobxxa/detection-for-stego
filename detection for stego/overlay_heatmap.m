function out = overlay_heatmap(gray01, score01, alpha)
    % 将 score01 映射到 hot 伪彩并叠加到灰度图（0..1）
    if ~isa(gray01,'double'), gray01 = im2double(gray01); end
    gray01 = repmat(gray01, [1 1 3]);
    cmap = hot(256);
    idx  = uint16(round(minmax_norm(score01)*255))+1;
    color = ind2rgb(idx, cmap);
    out = (1-alpha)*gray01 + alpha*color;
    out = max(0, min(1, out));
end
