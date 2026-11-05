function mapped_labels = bestMap(true_labels, predicted_labels)
%BESTMAP Match predicted cluster labels to true labels for ACC calculation.
% Uses dynamic programming for small label sets and a Hungarian assignment
% solver for larger sets, without requiring an extra toolbox.
true_labels = true_labels(:);
predicted_labels = predicted_labels(:);
if numel(true_labels) ~= numel(predicted_labels)
    error('The two label vectors must have the same number of elements.');
end
true_classes = unique(true_labels);
predicted_classes = unique(predicted_labels);
if isempty(true_classes) || isempty(predicted_classes)
    error('Label vectors must not be empty.');
end
n_true = numel(true_classes);
n_pred = numel(predicted_classes);
n = max(n_true, n_pred);
counts = zeros(n, n);
for i = 1:n_true
    for j = 1:n_pred
        counts(i,j) = sum(true_labels == true_classes(i) & ...
            predicted_labels == predicted_classes(j));
    end
end

if n > 20
    % Hungarian assignment is cubic in n and supports large class counts.
    % with 40 classes. The small-set branch below preserves earlier results.
    assignment = hungarian_max_assignment(counts);
else
    % Maximize overlap over one-to-one assignments using O(n*2^n) dynamic programming.
    state_count = 2^n;
    best_score = -inf(state_count, 1);
    parent_column = zeros(state_count, 1, 'uint32');
    best_score(1) = 0;
    for mask = 0:(state_count - 1)
        row = sum(bitget(uint64(mask), 1:n)) + 1;
        if row > n || ~isfinite(best_score(mask + 1))
            continue;
        end
        for col = 1:n
            if bitget(uint64(mask), col) == 0
                next_mask = bitset(uint64(mask), col, 1);
                next_index = double(next_mask) + 1;
                candidate = best_score(mask + 1) + counts(row, col);
                if candidate > best_score(next_index)
                    best_score(next_index) = candidate;
                    parent_column(next_index) = uint32(col);
                end
            end
        end
    end
    assignment = zeros(1, n);
    mask = uint64(state_count - 1);
    for row = n:-1:1
        col = double(parent_column(double(mask) + 1));
        assignment(row) = col;
        mask = bitset(mask, col, 0);
    end
end
predicted_to_true = zeros(1, n);
for row = 1:n
    predicted_to_true(assignment(row)) = row;
end

mapped_labels = zeros(size(predicted_labels), 'like', true_labels);
for j = 1:n_pred
    i = predicted_to_true(j);
    if i < 1 || i > n_true
        [~, i] = max(counts(1:n_true, j));
    end
    mapped_labels(predicted_labels == predicted_classes(j)) = true_classes(i);
end
end

function assignment = hungarian_max_assignment(weights)
%HUNGARIAN_MAX_ASSIGNMENT Row-to-column maximum weight assignment.
n = size(weights, 1);
cost = max(weights(:)) - weights;
u = zeros(n + 1, 1);
v = zeros(n + 1, 1);
p = zeros(n + 1, 1);
way = zeros(n + 1, 1);
for i = 1:n
    p(1) = i;
    j0 = 1;
    minv = inf(n + 1, 1);
    used = false(n + 1, 1);
    while true
        used(j0) = true;
        i0 = p(j0);
        delta = inf;
        j1 = 0;
        for j = 2:(n + 1)
            if ~used(j)
                cur = cost(i0, j - 1) - u(i0 + 1) - v(j);
                if cur < minv(j)
                    minv(j) = cur;
                    way(j) = j0;
                end
                if minv(j) < delta
                    delta = minv(j);
                    j1 = j;
                end
            end
        end
        for j = 1:(n + 1)
            if used(j)
                u(p(j) + 1) = u(p(j) + 1) + delta;
                v(j) = v(j) - delta;
            else
                minv(j) = minv(j) - delta;
            end
        end
        j0 = j1;
        if p(j0) == 0
            break;
        end
    end
    while true
        j1 = way(j0);
        p(j0) = p(j1);
        j0 = j1;
        if j0 == 1
            break;
        end
    end
end
assignment = zeros(1, n);
for j = 2:(n + 1)
    if p(j) > 0
        assignment(p(j)) = j - 1;
    end
end
end
