function global_W = ServerAggregate(local_W, client_weights, V, K)
%SERVERAGGREGATE Sample-count-weighted aggregation without label-derived weights.
client_weights = double(client_weights(:)');
if numel(client_weights) ~= K || any(~isfinite(client_weights)) || sum(client_weights) <= 0
    client_weights = ones(1, K);
end
agg_weights = client_weights / sum(client_weights);
for v = 1:V
    W_sum = zeros(size(local_W{1}{v}));
    for k = 1:K
        W_sum = W_sum + agg_weights(k) * local_W{k}{v};
    end
    global_W{v} = W_sum;
end
end