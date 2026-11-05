function [global_W, mean_local_obj] = SecureServerAggregate(masked_uploads, client_weights, V, K)
%SECURESERVERAGGREGATE Recover only weighted model and summed objective values.
% Honest-but-curious fixed-roster aggregation only; dropout recovery and
% malicious-client defenses are not implemented.
if numel(masked_uploads) ~= K || any(cellfun(@isempty, masked_uploads))
    error('SecureServerAggregate:MissingClient', ...
        'All registered clients must upload in this fixed-participant protocol.');
end
weights = double(client_weights(:)');
if numel(weights) ~= K || any(~isfinite(weights)) || any(weights <= 0) || ...
        any(weights ~= floor(weights))
    error('SecureServerAggregate:InvalidWeights', ...
        'Public sample counts must be finite and positive.');
end

first = masked_uploads{1};
q = first.modulus;
scale = first.scale;
D = first.dimension;
model_D = first.model_dimension;
if D ~= model_D + 1
    error('SecureServerAggregate:InvalidPayloadDimension', ...
        'Payload must contain model parameters and one objective value.');
end
sum_mod = zeros(D, 1);
seen = false(1, K);
for k = 1:K
    u = masked_uploads{k};
    if u.client_id < 1 || u.client_id > K || seen(u.client_id)
        error('SecureServerAggregate:DuplicateClient', ...
            'Uploads must contain each registered client exactly once.');
    end
    if u.sample_count ~= weights(u.client_id)
        error('SecureServerAggregate:SampleCountMismatch', ...
            'Public sample count does not match the client-weighted upload.');
    end
    if u.round_id ~= first.round_id || u.dimension ~= D || ...
            u.model_dimension ~= model_D || ...
            u.scale ~= scale || u.modulus ~= q || numel(u.values) ~= D
        error('SecureServerAggregate:IncompatibleUpload', ...
            'Upload metadata does not match the active aggregation round.');
    end
    seen(u.client_id) = true;
    sum_mod = mod(sum_mod + u.values(:), q);
end
if ~all(seen)
    error('SecureServerAggregate:MissingClient', 'An expected client upload is absent.');
end

% Decode the signed fixed-point sum; the client-side bound prevents wraparound.
signed_sum = sum_mod;
signed_sum(sum_mod > q / 2) = sum_mod(sum_mod > q / 2) - q;
weighted_average = signed_sum(1:model_D) ./ (scale * sum(weights));
mean_local_obj = signed_sum(model_D + 1) ./ (scale * K);

cursor = 0;
global_W = cell(1, V);
for v = 1:V
    dims = first.shapes{v};
    count = prod(dims);
    global_W{v} = reshape(weighted_average(cursor + (1:count)), dims);
    cursor = cursor + count;
end
if cursor ~= model_D
    error('SecureServerAggregate:ShapeMismatch', 'View shapes do not sum to payload dimension.');
end
end
