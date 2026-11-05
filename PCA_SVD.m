function [score, coeff, mu] = PCA_SVD(X, numComponents)
X = double(X);
if any(~isfinite(X(:)))
    error('PCA_SVD expects finite input; fill values and restore missing masks afterwards.');
end
mu = mean(X, 1);
Xc = bsxfun(@minus, X, mu);
[U, S, V] = svd(Xc, 'econ');
q = min([numComponents, size(U, 2), size(V, 2)]);
score = U(:, 1:q) * S(1:q, 1:q);
coeff = V(:, 1:q);
end