function [global_W, last_mean_obj, Xhat_global, history] = FederatedFeatureSelection( ...
    X, Y, c, V, K, client_indices, options, max_comm_rounds)
%FEDERATEDFEATURESELECTION Fed-UIMUFS client optimization and RAFA rounds.
% Clients call ClientCoupledOptimization; the server aggregates selectors.
% The sample-count FedAvg branch is retained for ablation experiments.
% With options.use_rafa=true, clients locally align selectors for RAFA.
% Enabling secure_aggregation masks the uploaded sufficient statistics;
% otherwise the same RAFA sums are aggregated directly. Labels are unused.
% Set federated_early_stopping=false to execute the entire round budget;
% first_tolerance_round still records when the usual threshold is first met.
if nargin < 8 || isempty(max_comm_rounds), max_comm_rounds = 1; end
if nargin < 7 || isempty(options), options = struct(); end
if numel(Y) ~= size(X{1}, 2)
    error('FederatedFeatureSelection:SampleLabelMismatch', 'Y is retained for API compatibility only.');
end

for v = 1:V
    global_W{v} = zeros(size(X{v}, 1), c);
end
last_mean_obj = inf;
local_Xhat = cell(1, K);
if ~isfield(options, 'fedprox_mu') || isempty(options.fedprox_mu)
    options.fedprox_mu = 0;
end
use_rafa = isfield(options, 'use_rafa') && logical(options.use_rafa);
parallel_clients = logical(get_option(options, 'parallel_clients', false)) && K > 1;
use_secure_aggregation = isfield(options, 'secure_aggregation') && ...
    logical(options.secure_aggregation) && K > 1;
if use_rafa && K < 2
    warning('FederatedFeatureSelection:RAFASingleClient', ...
        'RAFA needs multiple clients; a single-client run reduces to the local selector.');
end
if isfield(options, 'secure_aggregation') && options.secure_aggregation && K == 1
    warning('FederatedFeatureSelection:SecureAggregationSingleClient', ...
        'Secure aggregation needs at least two clients; using a local direct update for K=1.');
end
if use_secure_aggregation
    secure_state = SecureAggregationSetup(K);
else
    secure_state = [];
end

sample_counts = cellfun(@numel, client_indices);
n_rounds = max_comm_rounds;
if K == 1
    n_rounds = 1;
end
if n_rounds < 1 || n_rounds ~= floor(n_rounds)
    error('FederatedFeatureSelection:InvalidRoundCount', 'max_comm_rounds must be a positive integer.');
end
history.round = (1:n_rounds)';
history.mean_objective = nan(n_rounds, 1);
history.relative_change = nan(n_rounds, 1);
history.tolerance_met = false(n_rounds, 1);
history.first_tolerance_round = NaN;
history.converged = false;
history.stop_round = n_rounds;
history.stop_reason = 'maximum communication rounds reached';
history.client_sample_counts = sample_counts;
client_rng_states = cell(1, K);
if parallel_clients
    % Give each simulated client a reproducible stream that advances across rounds.
    parallel_seed = get_option(options, 'parallel_seed', ...
        get_option(options, 'sghbo_fitness_seed', 42));
    caller_rng_state = rng;
    for k = 1:K
        rng(parallel_seed + k, 'twister');
        client_rng_states{k} = rng;
    end
    rng(caller_rng_state);
end
record_local_trace = isfield(options, 'federated_record_local_trace') && ...
    logical(options.federated_record_local_trace);
if record_local_trace
    % Diagnostic scalar trajectories are retained by the simulation driver.
    history.local_trace = cell(K, n_rounds);
end
if use_rafa && use_secure_aggregation
    history.aggregation = 'secure RAFA';
elseif use_rafa
    history.aggregation = 'RAFA';
else
    history.aggregation = 'sample-count FedAvg';
end

tol = 1e-4;
if isfield(options, 'federated_tolerance') && ~isempty(options.federated_tolerance)
    tol = options.federated_tolerance;
end
early_stopping = get_option(options, 'federated_early_stopping', true);
history.early_stopping_enabled = logical(early_stopping);
history.parallel_clients = parallel_clients;
if ~early_stopping
    history.stop_reason = 'fixed communication budget; early stopping disabled';
end
zeta_F = get_option(options, 'rafa_zeta', 0.1);
nu_n = get_option(options, 'rafa_nu_n', 1);
nu_q = get_option(options, 'rafa_nu_q', 1);
nu_R = get_option(options, 'rafa_nu_R', 1);
nu_v = get_option(options, 'rafa_nu_v', 1);
rafa_epsilon = get_option(options, 'rafa_epsilon', 1e-8);

for comm = 1:n_rounds
    fprintf('--- Fed-UIMUFS communication round %d ---\n', comm);
    local_W = cell(1, K);
    local_obj = zeros(1, K);
    local_alpha = zeros(K, V);
    local_reliability = zeros(K, V);
    local_quality = zeros(K, 1);
    local_valid = false(K, V);

    client_inputs = cell(1, K);
    for k = 1:K
        client_inputs{k} = cell(1, V);
        for v = 1:V
            client_inputs{k}{v} = X{v}(:, client_indices{k});
        end
    end

    client_results = cell(1, K);
    if parallel_clients
        parfor k = 1:K
            client_results{k} = run_federated_client_round( ...
                client_inputs{k}, k, comm, K, V, c, options, global_W, ...
            use_rafa, record_local_trace, client_rng_states{k});
        end
    else
        for k = 1:K
            client_results{k} = run_federated_client_round( ...
                client_inputs{k}, k, comm, K, V, c, options, global_W, ...
            use_rafa, record_local_trace, client_rng_states{k});
        end
    end

    for k = 1:K
        client_result = client_results{k};
        if parallel_clients
            client_rng_states{k} = client_result.rng_state;
        end
        local_W{k} = client_result.W;
        local_Xhat{k} = client_result.Xhat;
        local_obj(k) = client_result.objective;
        local_alpha(k, :) = client_result.alpha;
        local_reliability(k, :) = client_result.reliability;
        local_quality(k) = client_result.quality;
        local_valid(k, :) = client_result.valid;
        if record_local_trace
            history.local_trace{k, comm} = client_result.trace;
        end
    end

    if use_rafa
        aligned_W = cell(1, K);
        for k = 1:K
            aligned_W{k} = cell(1, V);
            for v = 1:V
                if comm == 1
                    aligned_W{k}{v} = local_W{k}{v};
                else
                    aligned_W{k}{v} = procrustes_align(local_W{k}{v}, global_W{v});
                end
            end
        end

        client_R = sum(local_alpha .* local_reliability, 2);
        phi = zeros(K, V);
        for k = 1:K
            for v = 1:V
                if local_valid(k, v)
                    phi(k, v) = sample_counts(k)^nu_n * ...
                        (local_quality(k) + rafa_epsilon)^nu_q * ...
                        (client_R(k) + rafa_epsilon)^nu_R * ...
                        (local_reliability(k, v) + rafa_epsilon)^nu_v;
                end
            end
        end

        if use_secure_aggregation
            masked_uploads = cell(1, K);
            for k = 1:K
                masked_uploads{k} = SecureMaskUploadRAFA(aligned_W{k}, phi(k, :), ...
                    local_valid(k, :), local_obj(k), k, secure_state, comm);
            end
            [global_W, mean_obj] = SecureServerAggregateRAFA( ...
                masked_uploads, global_W, K, zeta_F);
        else
            global_W = rafa_direct_aggregate(aligned_W, phi, local_valid, global_W, zeta_F, V);
            mean_obj = mean(local_obj);
        end
    elseif use_secure_aggregation
        masked_uploads = cell(1, K);
        for k = 1:K
            masked_uploads{k} = SecureMaskUpload( ...
                local_W{k}, sample_counts(k), k, secure_state, comm, local_obj(k));
        end
        [global_W, mean_obj] = SecureServerAggregate(masked_uploads, sample_counts, V, K);
    else
        global_W = ServerAggregate(local_W, sample_counts, V, K);
        mean_obj = mean(local_obj);
    end

    history.mean_objective(comm) = mean_obj;
    if comm > 1
        history.relative_change(comm) = abs(mean_obj - last_mean_obj) / max(abs(last_mean_obj), 1);
    end
    last_mean_obj = mean_obj;
    history.tolerance_met(comm) = comm > 1 && history.relative_change(comm) < tol;
    if history.tolerance_met(comm) && isnan(history.first_tolerance_round)
        history.first_tolerance_round = comm;
    end
    if early_stopping && history.tolerance_met(comm)
        history.converged = true;
        history.stop_round = comm;
        history.stop_reason = 'relative mean-objective change below tolerance';
        disp('Fed-UIMUFS mean objective converged; stopping communication rounds.');
        break;
    end
end

history.round = history.round(1:comm);
history.mean_objective = history.mean_objective(1:comm);
history.relative_change = history.relative_change(1:comm);
history.tolerance_met = history.tolerance_met(1:comm);
if record_local_trace
    history.local_trace = history.local_trace(:, 1:comm);
end
if ~history.converged
    history.stop_round = comm;
end

Xhat_global = cell(1, V);
n = size(X{1}, 2);
for v = 1:V
    Xhat_global{v} = zeros(size(X{v}, 1), n);
end
for k = 1:K
    for v = 1:V
        Xhat_global{v}(:, client_indices{k}) = local_Xhat{k}{v};
    end
end
fprintf('Mean local objective %.8g after %d communication rounds (%s).\n', ...
    last_mean_obj, history.stop_round, history.stop_reason);
end

function [alpha, stats] = estimate_local_reliability(X_local, options)
% Capture imputation diagnostics while keeping the local optimization log concise.
standardization = get_option(options, 'imputation_standardization', 'feature_zscore');
imputation_options = struct('standardization', standardization);
evalc('[~, alpha, stats] = ImputeUnbalancedViews(X_local, imputation_options);');
alpha = double(alpha(:)');
alpha = max(alpha, 0);
alpha = alpha ./ (sum(alpha) + eps);
end

function W_aligned = procrustes_align(W, W_reference)
if ~isequal(size(W), size(W_reference))
    error('FederatedFeatureSelection:SelectorShapeMismatch', ...
        'Local and global selectors must have matching dimensions for alignment.');
end
C = W' * W_reference;
[U, ~, Vsvd] = svd(C, 'econ');
W_aligned = W * (U * Vsvd');
end

function global_W = rafa_direct_aggregate(aligned_W, phi, valid, previous_W, zeta_F, V)
for v = 1:V
    active = find(valid(:, v));
    if isempty(active)
        global_W{v} = previous_W{v};
        continue;
    end
    evidence = phi(active, v);
    evidence_sum = sum(evidence);
    uniform_mean = zeros(size(aligned_W{active(1)}{v}));
    weighted_sum = zeros(size(uniform_mean));
    for ii = 1:numel(active)
        k = active(ii);
        uniform_mean = uniform_mean + aligned_W{k}{v};
        weighted_sum = weighted_sum + evidence(ii) .* aligned_W{k}{v};
    end
    uniform_mean = uniform_mean ./ numel(active);
    if evidence_sum > 0
        evidence_mean = weighted_sum ./ evidence_sum;
    else
        evidence_mean = uniform_mean;
    end
    global_W{v} = (1 - zeta_F) .* evidence_mean + zeta_F .* uniform_mean;
end
end

function result = run_federated_client_round( ...
    X_local, k, comm, K, V, c, options, global_W, use_rafa, record_local_trace, client_rng_state)
% Run one client's local work; this helper can execute on a parfor worker.
local_options = options;
alpha = zeros(1, V);
reliability = zeros(1, V);
quality = 0;
valid = false(1, V);

if logical(get_option(options, 'parallel_clients', false))
    % Restore that client's stream; the updated state is returned for the next round.
    rng(client_rng_state);
end

if use_rafa
    [alpha, reliability_stats] = estimate_local_reliability(X_local, options);
    reliability = reliability_stats.mean_reliability;
    for v = 1:V
        valid(v) = any(~all(isnan(X_local{v}), 1));
    end
    local_options.view_weights = alpha;
elseif isfield(options, 'view_weights')
    local_options.view_weights = options.view_weights;
end

if K > 1 && comm > 1 && local_options.fedprox_mu > 0
    local_options.fedprox_center = global_W;
else
    local_options.fedprox_mu = 0;
end

if record_local_trace
    [W, Xhat, objective, trace] = ...
        ClientCoupledOptimization(X_local, c, local_options);
else
    [W, Xhat, objective] = ...
        ClientCoupledOptimization(X_local, c, local_options);
    trace = [];
end

if use_rafa
    qseed = get_option(options, 'rafa_stability_seed', 42) + 1009 * comm + 9176 * k;
    quality = EstimateClusteringStability(Xhat, W, alpha, c, qseed);
end

result = struct('W',{W},'Xhat',{Xhat},'objective',objective, ...
    'alpha',alpha,'reliability',reliability,'quality',quality, ...
    'valid',valid,'trace',trace,'rng_state',rng);
end

function value = get_option(options, field, fallback)
if isfield(options, field) && ~isempty(options.(field))
    value = options.(field);
else
    value = fallback;
end
end

