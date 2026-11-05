function q = EstimateClusteringStability(Xhat, W, view_weights, c, seed)
%ESTIMATECLUSTERINGSTABILITY Label-free bootstrap stability in [0,1].
% Each bootstrap subsample is clustered independently; adjusted Rand indices
% are computed on pairwise overlaps, so no ground-truth labels are used.
if nargin < 5 || isempty(seed), seed = 42; end
if nargin < 4 || isempty(c), c = 2; end

embedding = BuildClusterEmbedding(Xhat, W, view_weights);
n = size(embedding, 1);
if n <= c
    q = 0;
    return;
end

previous_rng = rng;
restore_rng = onCleanup(@() rng(previous_rng)); %#ok<NASGU>
rng(seed, 'twister');
n_boot = 8;
sample_size = min(n, max(c + 1, round(0.8 * n)));
indices = cell(n_boot, 1);
for b = 1:n_boot
    indices{b} = randperm(n, sample_size);
end

labels = cell(n_boot, 1);
for b = 1:n_boot
    [labels{b}, ~] = KMeansSimple(embedding(indices{b}, :), c, 2, 100, seed + b);
end

stability = [];
for a = 1:n_boot-1
    for b = a+1:n_boot
        [~, ia, ib] = intersect(indices{a}, indices{b});
        if numel(ia) >= max(c + 1, 3)
            stability(end+1, 1) = ARI(labels{a}(ia), labels{b}(ib)); %#ok<AGROW>
        end
    end
end
if isempty(stability)
    q = 0;
else
    q = min(max(mean(stability), 0), 1);
end
end
