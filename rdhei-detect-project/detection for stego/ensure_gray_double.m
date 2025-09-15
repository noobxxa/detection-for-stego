function I = ensure_gray_double(I)
    if size(I,3)==3, I = rgb2gray(I); end
    if ~isa(I,'double'), I = im2double(I); end
    I = max(0, min(1, I));
end
