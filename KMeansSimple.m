function [labels, bestCost] = KMeansSimple(X, k, replicates, maxIter, seed)
%KMEANSSIMPLE Toolbox-free k-means++ with Lloyd refinement.
if nargin < 3 || isempty(replicates), replicates = 10; end
if nargin < 4 || isempty(maxIter), maxIter = 200; end
if nargin >= 5 && ~isempty(seed), rng(seed, 'twister'); end
X = double(X);
X(~isfinite(X)) = 0;
[n, d] = size(X);
if k < 1 || k > n
    error('KMeansSimple requires 1 <= k <= number of samples.');
end
bestCost = inf;
labels = ones(n, 1);
for rep = 1:replicates
    C = zeros(k, d);
    C(1, :) = X(randi(n), :);
    minD2 = local_sqdist(X, C(1, :));
    for j = 2:k
        w = max(minD2, 0);
        total = sum(w);
        if total <= eps
            idx = randi(n);
        else
            r = rand * total;
            idx = find(cumsum(w) >= r, 1, 'first');
            if isempty(idx), idx = randi(n); end
        end
        C(j, :) = X(idx, :);
        minD2 = min(minD2, local_sqdist(X, C(j, :)));
    end

    oldLabels = zeros(n, 1);
    for iter = 1:maxIter
        D2 = max(bsxfun(@plus, sum(X .* X, 2), sum(C .* C, 2)') - 2 * X * C', 0);
        [minD2, currentLabels] = min(D2, [], 2);
        if isequal(currentLabels, oldLabels), break; end
        oldLabels = currentLabels;
        for j = 1:k
            members = currentLabels == j;
            if any(members)
                C(j, :) = mean(X(members, :), 1);
            else
                [~, farthest] = max(minD2);
                C(j, :) = X(farthest, :);
            end
        end
    end
    D2 = max(bsxfun(@plus, sum(X .* X, 2), sum(C .* C, 2)') - 2 * X * C', 0);
    [minD2, currentLabels] = min(D2, [], 2);
    cost = sum(minD2);
    if cost < bestCost
        bestCost = cost;
        labels = currentLabels;
    end
end
end

function d2 = local_sqdist(X, center)
d2 = max(sum(X .* X, 2) + sum(center .* center, 2) - 2 * X * center', 0);
end