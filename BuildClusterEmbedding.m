function feaSet = BuildClusterEmbedding(X, W, view_weights)
%BUILDCLUSTEREMBEDDING Select features by selector row-energy importance x view weight.

V = numel(X);
if nargin < 3 || isempty(view_weights)
    view_weights = ones(1, V);
end
view_weights = double(view_weights(:))';
view_weights = max(view_weights, 0);
view_weights = view_weights / (sum(view_weights) + eps);

Xall = [];
score = [];
for v = 1:V
    Xv = X{v};
    Xv(~isfinite(Xv)) = 0;
    Xall = [Xall; Xv]; %#ok<AGROW>
    if ~isempty(W) && numel(W) >= v && ~isempty(W{v})
        sv = sum(W{v} .* W{v}, 2) * view_weights(v);
    else
        sv = var(Xv, 0, 2) * view_weights(v);
    end
    score = [score; sv]; %#ok<AGROW>
end

d = size(Xall, 1);
keep = max(30, round(0.5 * d));
keep = min(keep, d);
[~, idx] = sort(score, 'descend');
feaSet = Xall(idx(1:keep), :)';
feaSet = feaSet ./ (sqrt(sum(feaSet.^2, 2)) + eps);
end
