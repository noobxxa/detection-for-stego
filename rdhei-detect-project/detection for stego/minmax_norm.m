function Y = minmax_norm(X)
    X = double(X);
    mn = min(X(:)); mx = max(X(:));
    if mx>mn
        Y = (X-mn)/(mx-mn);
    else
        Y = zeros(size(X));
    end
end