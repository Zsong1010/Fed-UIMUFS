function client_indices = FederatedPartition(n, K)
idx_all = randperm(n);
client_indices = cell(1, K);
client_size = floor(n / K);
for k = 1:K
    if k < K
        client_indices{k} = idx_all((k-1)*client_size+1 : k*client_size);
    else
        client_indices{k} = idx_all((k-1)*client_size+1 : end);
    end
end
end 