function [Xhat, view_weights, stats] = ImputeUnbalancedViews(X, opts)
%IMPUTEUNBALANCEDVIEWS Impute missing views with unbalanced missing rates.
% Combines:
%   1) observed-only z-score
%   2) cross-view ridge maps (global)
%   3) cross-view kNN graph reconstruction (local)
%   4) iterative consensus refinement
% Observed entries are never overwritten.
%
% view_weights: quality weights from completeness x reconstruction reliability.

if nargin < 2 || isempty(opts), opts = struct(); end
if ~isfield(opts, 'ridge') || isempty(opts.ridge), opts.ridge = 1e-2; end
if ~isfield(opts, 'knn') || isempty(opts.knn), opts.knn = 7; end
if ~isfield(opts, 'n_iter') || isempty(opts.n_iter), opts.n_iter = 2; end
if ~isfield(opts, 'standardization') || isempty(opts.standardization)
    opts.standardization = 'feature_zscore';
end

V = numel(X);
n = size(X{1}, 2);
miss = false(V, n);
for v = 1:V
    miss(v, :) = all(isnan(X{v}), 1) | isnan(X{v}(1, :));
end
rates = mean(miss, 2)';

% Preserve the legacy observed-only feature z-score by default. Sparse text
% vectors may instead use 'none' here and receive column L2 normalization
% in the caller, avoiding densification of zero-valued word features.
Xz = X;
switch lower(char(opts.standardization))
    case 'feature_zscore'
        for v = 1:V
            obs = ~miss(v, :);
            if ~any(obs)
                continue;
            end
            mu = mean(X{v}(:, obs), 2);
            sd = std(X{v}(:, obs), 0, 2);
            sd(sd < 1e-8) = 1;
            tmp = X{v};
            tmp(:, obs) = (tmp(:, obs) - mu) ./ sd;
            Xz{v} = tmp;
        end
    case 'none'
        % Keep observed values on their input scale; callers may apply
        % sample-wise L2 normalization without destroying sparsity.
    otherwise
        error('ImputeUnbalancedViews:UnknownStandardization', ...
            'standardization must be ''feature_zscore'' or ''none''.');
end

Xhat = Xz;
for v = 1:V
    obs = ~miss(v, :);
    if any(obs)
        mu_v = mean(Xz{v}(:, obs), 2);
    else
        mu_v = zeros(size(Xz{v}, 1), 1);
    end
    tmp = Xz{v};
    tmp(:, miss(v, :)) = repmat(mu_v, 1, nnz(miss(v, :)));
    Xhat{v} = tmp;
end

rel = ones(V, n);
rel(miss) = 0.3;

for it = 1:opts.n_iter
    Xhat = ridge_fill(Xhat, Xz, miss, opts.ridge);
    [Xhat, rel_knn] = knn_fill(Xhat, miss, opts.knn);
    rel(miss) = max(rel(miss), rel_knn(miss));
end

% Keep original observed values
for v = 1:V
    obs = ~miss(v, :);
    Xhat{v}(:, obs) = Xz{v}(:, obs);
    rel(v, obs) = 1;
end

% Completeness x mean reliability; sqrt softens extreme 0.2/0.6 imbalance
mean_rel = mean(rel, 2)';
complete = max(1 - rates, 1e-3);
view_weights = (sqrt(complete) .* max(mean_rel, 1e-3));
view_weights = view_weights / sum(view_weights);

stats = struct();
stats.missing_rate_before = rates;
stats.missing_rate_after = zeros(1, V);
stats.mean_reliability = mean_rel;
for v = 1:V
    stats.missing_rate_after(v) = mean(all(isnan(Xhat{v}), 1));
end

fprintf('Unbalanced missing rates before imputation: %s\n', mat2str(rates, 4));
fprintf('Missing rates after imputation: %s\n', mat2str(stats.missing_rate_after, 4));
fprintf('View reliability: %s\n', mat2str(mean_rel, 4));
fprintf('View weights: %s\n', mat2str(view_weights, 4));
end

function Xhat = ridge_fill(Xhat, Xobs, miss, ridge)
V = numel(Xhat);
for v = 1:V
    dv = size(Xhat{v}, 1);
    obs_v = ~miss(v, :);
    if ~any(obs_v)
        continue;
    end
    mu_v = mean(Xobs{v}(:, obs_v), 2);
    acc = zeros(dv, size(Xhat{v}, 2));
    cnt = zeros(1, size(Xhat{v}, 2));
    for u = 1:V
        if u == v
            continue;
        end
        both = obs_v & ~miss(u, :);
        tgt = miss(v, :) & ~miss(u, :);
        if nnz(both) < 10 || ~any(tgt)
            continue;
        end
        Xu = Xobs{u}(:, both);
        Xv_b = Xobs{v}(:, both);
        mu_u = mean(Xu, 2);
        Xu_c = Xu - mu_u;
        Xv_c = Xv_b - mu_v;
        du = size(Xu, 1);
        n_both = size(Xu_c, 2);
        Xu_m = Xhat{u}(:, tgt) - mu_u;
        if n_both <= du
            pred = Xv_c * ((Xu_c' * Xu_c + ridge * eye(n_both)) \ (Xu_c' * Xu_m));
        else
            Map = (Xv_c * Xu_c') / (Xu_c * Xu_c' + ridge * eye(du));
            pred = Map * Xu_m;
        end
        pred = pred + mu_v;
        acc(:, tgt) = acc(:, tgt) + pred;
        cnt(tgt) = cnt(tgt) + 1;
    end
    fill = miss(v, :) & (cnt > 0);
    if any(fill)
        Xhat{v}(:, fill) = acc(:, fill) ./ cnt(fill);
    end
end
end

function [Xhat, rel] = knn_fill(Xhat, miss, k)
V = numel(Xhat);
n = size(Xhat{1}, 2);
rel = zeros(V, n);

% Consensus search space: column-normalized concatenation
F = [];
for v = 1:V
    Xv = Xhat{v};
    nrm = sqrt(sum(Xv.^2, 1)) + eps;
    F = [F; Xv ./ nrm]; %#ok<AGROW>
end
F = F';
F = F ./ (sqrt(sum(F.^2, 2)) + eps);

for v = 1:V
    obs = find(~miss(v, :));
    mis = find(miss(v, :));
    if isempty(mis) || numel(obs) < 3
        continue;
    end
    kk = min(k, numel(obs));
    [nn, dist] = knnsearch_rows(F(obs, :), F(mis, :), kk);
    md = median(dist(:)) + eps;
    w = exp(-dist ./ md);
    w = w ./ (sum(w, 2) + eps);
    Xobs = Xhat{v}(:, obs);
    rec = zeros(size(Xhat{v}, 1), numel(mis));
    for j = 1:numel(mis)
        rec(:, j) = Xobs(:, nn(j, :)) * w(j, :)';
    end
    Xhat{v}(:, mis) = rec;
    rel(v, mis) = max(w(:, 1)', 0.05);
    rel(v, obs) = 1;
end
end
function [nn, dist] = knnsearch_rows(reference, query, k)
% Exact Euclidean kNN fallback using base MATLAB matrix operations.
d2 = max(bsxfun(@plus, sum(reference .* reference, 2), ...
    sum(query .* query, 2)') - 2 * reference * query', 0);
[sortedD2, sortedIdx] = sort(d2, 1, 'ascend');
k = min(k, size(reference, 1));
nn = sortedIdx(1:k, :)';
dist = sqrt(sortedD2(1:k, :))';
end
