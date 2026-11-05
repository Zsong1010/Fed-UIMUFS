function [global_W, mean_obj, sums] = SecureServerAggregateRAFA(uploads, previous_W, K, zeta_F)
%SECURESERVERAGGREGATERAFA Recover only masked RAFA sufficient statistics.
if numel(uploads) ~= K || any(cellfun(@isempty, uploads))
    error('SecureServerAggregateRAFA:MissingClient', 'Every registered client must upload.');
end
if K < 2 || zeta_F < 0 || zeta_F > 1
    error('SecureServerAggregateRAFA:InvalidConfiguration', 'Invalid K or smoothing coefficient.');
end

first = uploads{1};
q = first.modulus;
scale = first.scale;
D = first.dimension;
if D < 1 || isempty(first.shapes)
    error('SecureServerAggregateRAFA:InvalidPayload', 'Empty RAFA payload.');
end
sum_mod = zeros(D, 1);
seen = false(1, K);
for k = 1:K
    u = uploads{k};
    if u.client_id < 1 || u.client_id > K || seen(u.client_id)
        error('SecureServerAggregateRAFA:DuplicateClient', 'Uploads must include each client once.');
    end
    if u.round_id ~= first.round_id || u.dimension ~= D || ...
            u.scale ~= scale || u.modulus ~= q || numel(u.values) ~= D || ...
            ~isequal(u.shapes, first.shapes)
        error('SecureServerAggregateRAFA:IncompatibleUpload', 'Upload metadata do not match.');
    end
    seen(u.client_id) = true;
    sum_mod = mod(sum_mod + u.values(:), q);
end
if ~all(seen)
    error('SecureServerAggregateRAFA:MissingClient', 'An expected client upload is absent.');
end

signed_sum = sum_mod;
signed_sum(sum_mod > q / 2) = signed_sum(sum_mod > q / 2) - q;
sums = signed_sum ./ scale;
V = numel(first.shapes);
global_W = cell(1, V);
cursor = 1;
for v = 1:V
    shape = first.shapes{v};
    count = prod(shape);
    weighted_sum = reshape(sums(cursor:cursor + count - 1), shape);
    cursor = cursor + count;
    phi_sum = sums(cursor);
    cursor = cursor + 1;
    uniform_sum = reshape(sums(cursor:cursor + count - 1), shape);
    cursor = cursor + count;
    valid_count = round(sums(cursor));
    cursor = cursor + 1;

    if valid_count == 0
        global_W{v} = previous_W{v};
    else
        if phi_sum > 0
            weighted_mean = weighted_sum ./ phi_sum;
        else
            weighted_mean = uniform_sum ./ valid_count;
        end
        global_W{v} = (1 - zeta_F) .* weighted_mean + ...
            zeta_F .* (uniform_sum ./ valid_count);
    end
end
if cursor ~= D
    error('SecureServerAggregateRAFA:ShapeMismatch', 'Payload layout does not match selectors.');
end
mean_obj = sums(end) ./ K;
sums = struct('payload_sum', sums(1:end-1), 'round_id', first.round_id, ...
    'mean_objective', mean_obj);
end
