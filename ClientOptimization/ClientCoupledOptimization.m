function [W, Xhat, objval, trace] = ClientCoupledOptimization(X, c, options)
%CLIENTCOUPLEDOPTIMIZATION Client-side RGTAL--SGHBO coupled optimization.
% X{v}: d_v-by-n local view; missing samples are NaN columns.
% c: number of clusters, also used as the latent and anchor dimension.
% W: local view selectors. Xhat: completed local views.
% objval: the final local RGTAL objective, including FedProx when enabled.
% Optional trace: raw RGTAL objectives (including initialization) and SGHBO
% elite fitness/objective values. Logging does not change optimizer updates.
%
% With options.use_sghbo=true, RGTAL initializes selectors and SGHBO refines
% their continuous coefficients. Each candidate fitness evaluation optimizes
% the remaining RGTAL variables with that candidate selector fixed.
% options.fixed_W evaluates a supplied selector through the same RGTAL
% updates; use_sghbo=false selects the analytical RGTAL ablation.
% options.fedprox_center supplies the previous global selectors when used.
% options.use_legacy_hbo enables the historical HBO comparison branch.
% The analytical solver is adapted from TIME-FS; see the source citation
% in the project README. SGHBO and the federated extensions are local changes.

trace = struct();
record_trace = nargout > 3;
options.r = c; % Set the number of anchors to c, the number of clusters.
options.k = c; % Set the latent subspace dimension to c, the number of clusters.
V = size(X,2); % Number of views.
[~,n] = size(X{1}); % Number of samples.
if ~isfield(options, 'view_weights') || isempty(options.view_weights)
    options.view_weights = ones(1, V);
end
options.view_weights = double(options.view_weights(:))';
if numel(options.view_weights) ~= V
    options.view_weights = ones(1, V);
end
options.view_weights = max(options.view_weights, 0);
options.view_weights = options.view_weights / (sum(options.view_weights) + eps);
if ~isfield(options, 'use_legacy_hbo')
    options.use_legacy_hbo = false;
end
if isfield(options, 'fixed_W') && ~isempty(options.fixed_W)
    [W, Xhat, objval] = rgtal_optimize_given_selector(X, c, options, options.fixed_W);
    trace.mode = 'fixed selector evaluation';
    return;
end
if isfield(options, 'use_sghbo') && options.use_sghbo
    if record_trace
        [W, Xhat, objval, trace] = sghbo_refine_selectors(X, c, options);
    else
        [W, Xhat, objval] = sghbo_refine_selectors(X, c, options);
    end
    return;
end
% ========== Historical HBO comparison branch ==========
% Parse hot-start initialization, shrinking local search, winner acceptance,
% elitist retention, and feature-concentration fitness.
% HBO update adapted from: https://github.com/IBRAlShourbaji/HBO_Optimizer
if isfield(options, 'use_legacy_hbo') && options.use_legacy_hbo
    d_list = zeros(1, V);
    w_dim = 0;
    for v = 1:V
        d_list(v) = size(X{v},1);
        w_dim = w_dim + d_list(v)*options.k;
    end
    num_particles = 15;
    max_ext_iter = 10;
    n_elite = max(2, floor(num_particles / 3));

    % --- Hot-start initialization ---
    [W0, ~] = rgtal_optimize(X, c, options);
    w0 = [];
    for v = 1:V
        w0 = [w0, W0{v}(:)']; %#ok<AGROW>
    end
    rad = max(0.2, 0.5 * std(w0));
    lb = min(w0) - 2 * rad;
    ub = max(w0) + 2 * rad;
    if lb >= ub
        lb = -1; ub = 1;
    end

    swarm = repmat(w0, num_particles, 1);
    swarm(1, :) = w0;
    for i = 2:n_elite
        swarm(i, :) = w0 + 0.15 * rad * randn(1, w_dim);
    end
    for i = (n_elite+1):num_particles
        swarm(i, :) = w0 + rad * randn(1, w_dim);
    end
    swarm = min(max(swarm, lb), ub);
    swarm_prev = swarm + 0.05 * rad * randn(size(swarm));
    swarm_prev = min(max(swarm_prev, lb), ub);

    GSR = zeros(num_particles, w_dim);
    GSRP = zeros(num_particles, w_dim);
    Fitness = inf(num_particles, 1);
    pbest = swarm;
    pbest_fitness = inf(num_particles, 1);
    gbest = w0;
    gbest_fitness = inf;

    for iter = 1:max_ext_iter
        shrink = 0.5 + 0.5 * (1 - (iter-1) / max_ext_iter);
        cur_lb = gbest - shrink * 2 * rad;
        cur_ub = gbest + shrink * 2 * rad;
        cur_lb = max(cur_lb, lb);
        cur_ub = min(cur_ub, ub);

        fitvals = inf(num_particles, 1);
        parfor i = 1:num_particles
            fitvals(i) = legacy_hbo_fitness(swarm(i,:), X, c, options, d_list, V);
        end
        Fitness = fitvals;

        for i = 1:num_particles
            if Fitness(i) < pbest_fitness(i)
                pbest_fitness(i) = Fitness(i);
                pbest(i, :) = swarm(i, :);
            end
            if Fitness(i) < gbest_fitness
                gbest_fitness = Fitness(i);
                gbest = swarm(i, :);
            end
        end

        % Propose new positions using only global-best guidance, without the
        % three-body interaction terms.
        GSRP = GSR;
        swarm_old = swarm;
        [cand, GSR] = legacy_hbo_update_population(swarm, swarm_prev, Fitness, ...
            GSR, GSRP, gbest, lb, ub);
        cand = min(max(cand, cur_lb), cur_ub);

        % Winner acceptance.
        cand_fit = inf(num_particles, 1);
        parfor i = 1:num_particles
            cand_fit(i) = legacy_hbo_fitness(cand(i,:), X, c, options, d_list, V);
        end
        for i = 1:num_particles
            if cand_fit(i) <= Fitness(i)
                swarm(i, :) = cand(i, :);
                Fitness(i) = cand_fit(i);
                swarm_prev(i, :) = swarm_old(i, :);
                if cand_fit(i) < pbest_fitness(i)
                    pbest_fitness(i) = cand_fit(i);
                    pbest(i, :) = cand(i, :);
                end
                if cand_fit(i) < gbest_fitness
                    gbest_fitness = cand_fit(i);
                    gbest = cand(i, :);
                end
            else
                swarm_prev(i, :) = 0.5 * swarm_old(i, :) + 0.5 * swarm(i, :);
            end
        end

        % Preserve the elite solution.
        swarm(1, :) = gbest;
        Fitness(1) = gbest_fitness;
        pbest(1, :) = gbest;
        pbest_fitness(1) = gbest_fitness;
    end

    % Decode the best W and refine it.
    W = cell(1, V);
    cnt = 0;
    for v = 1:V
        d = d_list(v);
        W{v} = reshape(gbest(cnt+1:cnt+d*options.k), d, options.k);
        cnt = cnt + d*options.k;
    end
    [W, Xhat, ~] = rgtal_optimize_given_selector(X, c, options, W);
    [W2, ~] = rgtal_optimize(X, c, options);
    for v = 1:V
        W{v} = 0.7 * W{v} + 0.3 * W2{v};
    end
    [W, Xhat, objval] = rgtal_optimize_given_selector(X, c, options, W);
    return;
end
% ========== Inner RGTAL analytical optimization ==========
if record_trace
    [W, Xhat, objval, trace.rgtal] = rgtal_optimize(X, c, options);
    trace.mode = 'analytical RGTAL';
else
    [W,Xhat,objval] = rgtal_optimize(X,c,options);
end
end

function [W, Xhat, objval, trace] = sghbo_refine_selectors(X, c, options)
% SGHBO selector refinement coupled to fixed-selector RGTAL evaluations.
record_trace = nargout > 3;
trace = struct();
V = numel(X);
if ~isfield(options, 'sghbo_particles'), options.sghbo_particles = 15; end
if ~isfield(options, 'sghbo_iterations'), options.sghbo_iterations = 10; end
if ~isfield(options, 'sghbo_eta'), options.sghbo_eta = 0.2; end
if ~isfield(options, 'sghbo_lambda_sg'), options.sghbo_lambda_sg = 0.05; end
if ~isfield(options, 'sghbo_con_weight'), options.sghbo_con_weight = 0.5; end
if ~isfield(options, 'sghbo_lambda_w'), options.sghbo_lambda_w = 0.25; end
if ~isfield(options, 'sghbo_beta_min'), options.sghbo_beta_min = 0.1; end
if ~isfield(options, 'sghbo_beta_max'), options.sghbo_beta_max = 0.9; end
if ~isfield(options, 'sghbo_theta'), options.sghbo_theta = 0.5; end
if ~isfield(options, 'sghbo_nu'), options.sghbo_nu = 1; end
if ~isfield(options, 'sghbo_xi'), options.sghbo_xi = 1; end
if ~isfield(options, 'sghbo_acm_damping'), options.sghbo_acm_damping = 0.25; end
if ~isfield(options, 'sghbo_fitness_seed'), options.sghbo_fitness_seed = 42; end
if ~isfield(options, 'sghbo_elite_count')
    options.sghbo_elite_count = max(2, floor(options.sghbo_particles / 3));
end

if record_trace
    [W0, ~, obj0, trace.rgtal] = rgtal_optimize(X, c, options);
    trace.mode = 'coupled RGTAL-SGHBO';
else
    [W0, ~, obj0] = rgtal_optimize(X, c, options);
end
d_list = cellfun(@(A) size(A, 1), X);
w_dim = sum(d_list .* options.k);
w0 = [];
for v = 1:V
    w0 = [w0, W0{v}(:)']; %#ok<AGROW>
end
rad = max(0.2, 0.5 * std(w0));
lb = min(w0) - 2 * rad;
ub = max(w0) + 2 * rad;
if lb >= ub, lb = -1; ub = 1; end

n_pop = options.sghbo_particles;
n_iter = options.sghbo_iterations;
n_elite = min(max(1, options.sghbo_elite_count), n_pop);
vw = ones(1, V);
if isfield(options, 'view_weights') && numel(options.view_weights) == V
    vw = max(double(options.view_weights(:))', 0);
end
vw = vw / (sum(vw) + eps);
rho = ((vw + eps) .^ options.sghbo_xi);
rho = rho / max(rho);
rho_dim = [];
for v = 1:V
    rho_dim = [rho_dim, repmat(rho(v), 1, d_list(v) * options.k)]; %#ok<AGROW>
end

swarm = repmat(w0, n_pop, 1);
for i = 2:n_elite
    swarm(i, :) = w0 + 0.15 * rad * randn(1, w_dim);
end
for i = (n_elite+1):n_pop
    swarm(i, :) = w0 + rad * randn(1, w_dim);
end
swarm = min(max(swarm, lb), ub);
for i = 2:n_pop
    swarm(i, :) = sghbo_project_rows(swarm(i, :), d_list, options.k, ...
        rho, options.sghbo_eta, options.sghbo_lambda_w);
end
swarm_prev = swarm + 0.05 * rad * randn(size(swarm));
swarm_prev = min(max(swarm_prev, lb), ub);
GSR = zeros(n_pop, w_dim);

lambda_con = options.sghbo_con_weight * max(abs(obj0), 1);
fitness = zeros(n_pop, 1);
candidate_objective = zeros(n_pop, 1);
for i = 1:n_pop
    [fitness(i), candidate_objective(i)] = sghbo_candidate_fitness(swarm(i, :), X, c, options, ...
        d_list, rho, lambda_con);
end

initial_best_fitness = min(fitness);
if record_trace
    trace.sghbo.iterations = (0:n_iter)';
    trace.sghbo.best_fitness = nan(n_iter + 1, 1);
    trace.sghbo.selected_objective = nan(n_iter + 1, 1);
    [trace.sghbo.best_fitness(1), initial_best_idx] = min(fitness);
    trace.sghbo.selected_objective(1) = candidate_objective(initial_best_idx);
end
for iter = 1:n_iter
    [~, best_idx] = min(fitness);
    gbest = swarm(best_idx, :);
    shrink = 0.5 + 0.5 * (1 - (iter - 1) / n_iter);
    cur_lb = max(lb, gbest - shrink * 2 * rad);
    cur_ub = min(ub, gbest + shrink * 2 * rad);

    old_swarm = swarm;
    old_fitness = fitness;
    old_objective = candidate_objective;
    old_prev = swarm_prev;
    old_GSR = GSR;
    [cand, GSR_new] = sghbo_update_population(old_swarm, old_prev, ...
        old_fitness, old_GSR, gbest, w0, rho_dim, options, iter, n_iter);

    for i = 1:n_pop
        cand(i, :) = sghbo_project_rows(cand(i, :), d_list, options.k, ...
            rho, options.sghbo_eta, options.sghbo_lambda_w);
        cand(i, :) = min(max(cand(i, :), cur_lb), cur_ub);
    end
    cand_fitness = zeros(n_pop, 1);
    cand_objective = zeros(n_pop, 1);
    for i = 1:n_pop
        [cand_fitness(i), cand_objective(i)] = sghbo_candidate_fitness(cand(i, :), X, c, ...
            options, d_list, rho, lambda_con);
    end

    % Parent-plus-offspring environmental selection preserves the elite set.
    pool_X = [old_swarm; cand];
    pool_F = [old_fitness; cand_fitness];
    pool_objective = [old_objective; cand_objective];
    [fitness_sorted, order] = sort(pool_F, 'ascend');
    order = order(1:n_pop);
    swarm = pool_X(order, :);
    fitness = fitness_sorted(1:n_pop);
    candidate_objective = pool_objective(order);
    if record_trace
        [trace.sghbo.best_fitness(iter + 1), best_trace_idx] = min(fitness);
        trace.sghbo.selected_objective(iter + 1) = candidate_objective(best_trace_idx);
    end
    swarm_prev = zeros(size(swarm));
    GSR = zeros(size(swarm));
    for j = 1:n_pop
        source = order(j);
        if source <= n_pop
            swarm_prev(j, :) = old_prev(source, :);
            GSR(j, :) = old_GSR(source, :);
        else
            parent = source - n_pop;
            swarm_prev(j, :) = old_swarm(parent, :);
            GSR(j, :) = GSR_new(parent, :);
        end
    end
end

[~, best_idx] = min(fitness);
best_vec = swarm(best_idx, :);
W = sghbo_decode(best_vec, d_list, options.k);
rng(options.sghbo_fitness_seed, 'twister');
[W, Xhat, objval] = rgtal_optimize_given_selector(X, c, options, W);
fprintf('SGHBO fitness %.6g -> %.6g; concentration weight %.4g; row threshold %.4g\n', ...
    initial_best_fitness, min(fitness), lambda_con, options.sghbo_lambda_w);
end

function [fitval, objval] = sghbo_candidate_fitness(w_vec, X, c, options, d_list, rho, lambda_con)
W_cell = sghbo_decode(w_vec, d_list, options.k);
try
    % Common random numbers make candidate scores comparable and repeatable.
    rng(options.sghbo_fitness_seed, 'twister');
    options.fixed_selector_objective_only = true;
    [Wfs, ~, objval] = rgtal_optimize_given_selector(X, c, options, W_cell);
    conc = 0;
    for v = 1:numel(Wfs)
        a = sqrt(sum(Wfs{v} .* Wfs{v}, 2));
        conc = conc + sum(a .* a) / (sum(a) ^ 2 + eps);
    end
    conc = conc / numel(Wfs);
    fitval = objval + lambda_con * (1 - conc);
    if ~isfinite(fitval), fitval = 1e12; end
catch
    fitval = 1e12;
    objval = Inf;
end
end

function W = sghbo_decode(w_vec, d_list, k)
W = cell(1, numel(d_list));
offset = 0;
for v = 1:numel(d_list)
    n = d_list(v) * k;
    W{v} = reshape(w_vec(offset + (1:n)), d_list(v), k);
    offset = offset + n;
end
end

function w_vec = sghbo_project_rows(w_vec, d_list, k, rho, eta_s, lambda_w)
W = sghbo_decode(w_vec, d_list, k);
for v = 1:numel(W)
    row_norm = sqrt(sum(W{v} .* W{v}, 2));
    gamma = eta_s * rho(v) * lambda_w;
    scale = max(0, 1 - gamma ./ (row_norm + eps));
    W{v} = W{v} .* scale;
end
w_vec = [];
for v = 1:numel(W)
    w_vec = [w_vec, W{v}(:)']; %#ok<AGROW>
end
end

function [Xnew, GSRnew] = sghbo_update_population(X, Xprev, Fitness, ...
    GSRprev, gbest, w0, rho_dim, options, iter, max_iter)
[N, dim] = size(X);
Xnew = zeros(size(X));
GSRnew = zeros(size(X));
Xmean = mean(X, 1);
base_beta = options.sghbo_beta_min + ...
    (options.sghbo_beta_max - options.sghbo_beta_min) * ...
    (1 - iter / max_iter) ^ options.sghbo_nu;
beta = base_beta .* (options.sghbo_theta + ...
    (1 - options.sghbo_theta) .* rho_dim);

for i = 1:N
    Xp = Xprev(i, :);
    XR = min(max(rand(1, dim), 0), 1);
    gsr = sghbo_gradient_operator(X(i, :), gbest, Xp, XR);
    gsr = gsr + options.sghbo_lambda_sg .* rho_dim .* (X(i, :) - w0);
    hbsr = X(i, :) - options.sghbo_eta .* ...
        (2 .* gsr - GSRprev(i, :)) + beta .* (X(i, :) - Xp);

    if N > 1
        a = randi(N - 1);
        if a >= i, a = a + 1; end
    else
        a = i;
    end
    RB = cumsum(sqrt(1 / max(dim, 1)) .* randn(1, dim));
    omega1 = -1 + 2 * rand;
    omega2 = -0.5 + rand;
    ebs = hbsr + omega1 .* (gbest - Xp) + ...
        omega2 .* RB .* (Xmean - X(a, :));

    den = ebs - 2 .* hbsr + gbest;
    small = abs(den) < 1e-8;
    den(small) = 1e-8 .* sign_nonzero(den(small));
    acm = gbest - (hbsr - gbest) .^ 2 ./ den;
    Xnew(i, :) = acm;
    GSRnew(i, :) = gsr;
end
end

function y = sign_nonzero(x)
y = sign(x);
y(y == 0) = 1;
end

function GSR = sghbo_gradient_operator(X, Xbest, Xprev, XR)
dim = size(X, 2);
S0 = rand(1, dim) .* (X - Xbest) ./ ...
    (((Xprev - rand .* X) .* (Xprev - rand .* Xbest)) + eps);
S1 = rand(1, dim) .* (2 .* X - Xprev - Xbest) ./ ...
    (((X - rand .* Xprev) .* (X - rand .* Xbest)) + eps);
S2 = rand(1, dim) .* (X - Xprev) ./ ...
    (((Xbest - rand .* Xprev) .* (Xbest - rand .* X)) + eps);
Y0 = rand(1, dim) .* (Xbest - rand .* Xprev);
Y1 = rand(1, dim) .* (Xbest - rand .* X);
Y2 = rand(1, dim) .* (Xbest - rand .* XR);
GSR = randn .* (Y0 .* S0 + Y1 .* S1 + Y2 .* S2);
end
function fitval = legacy_hbo_fitness(w_vec, X, c, options, d_list, V)
% Historical HBO fitness: local RGTAL objective / feature-energy concentration.
W_cell = cell(1, V);
cnt = 0;
for v = 1:V
    d = d_list(v);
    W_cell{v} = reshape(w_vec(cnt+1:cnt+d*options.k), d, options.k);
    cnt = cnt + d*options.k;
end
try
    [Wfs, ~, objval] = rgtal_optimize_given_selector(X, c, options, W_cell);
    ener = [];
    for v = 1:V
        ener = [ener; sum(Wfs{v} .* Wfs{v}, 2)]; %#ok<AGROW>
    end
    ener = sort(ener, 'descend');
    keep = max(1, round(0.3 * numel(ener)));
    conc = sum(ener(1:keep)) / (sum(ener) + eps); % Higher is better.
    fitval = objval / (conc + 0.05);
catch
    fitval = 1e12;
end
end

function mu = get_fedprox_mu(options)
mu = 0;
if isfield(options, 'fedprox_mu') && ~isempty(options.fedprox_mu)
    mu = max(0, options.fedprox_mu);
end
end
% RGTAL latent-variable and completion updates for a fixed selector.
function [W,Xhat,objval] = rgtal_optimize_given_selector(X,c,options,W)
if isfield(options,'fast_fixed_selector') && options.fast_fixed_selector
    [W,Xhat,objval] = rgtal_fixed_selector_grams(X,c,options,W);
    return;
end
% Optimize Z, A, M, P, and related variables while keeping the given W fixed.
V = size(X,2);
[~,n] = size(X{1});
observed=cell(1,V);
for v=1:V, observed{v}=~isnan(X{v}(1,:)); end
% Initialize the same RGTAL blocks while keeping the supplied W fixed.
% -------------- Initialize E-------------- %
G = cell(1,V); E = cell(1,V); Xhat = cell(1,V);
for v = 1:V
    [dv, n_local] = size(X{v});
    idx_v = find(ismissing(X{v}(1,:)));
    ns = numel(idx_v);
    G{v} = sparse(1:ns, idx_v, 1, ns, n_local);
    Xv = X{v};
    Xv(:, idx_v) = 0;
    if n_local - ns > 0
        E{v} = (sum(Xv, 2) / (n_local - ns)) * ones(1, ns);
    else
        E{v} = zeros(dv, ns);
    end
    X{v} = Xv;
    Xhat{v} = Xv + E{v} * G{v};
end
% -------------- Initialize Z -------------- %
Z = zeros(options.k,n,V);
for v = 1:V
    tmp = rand(n,options.k);
    [zr,~] = qr(tmp,0);
    zr = zr(:,1:options.k);
    Z(:,:,v) = zr';
end
% -------------- Initialize A M P -------------- %
A = rand(options.k,options.r);
P = rand(V,options.r);
M = updateM(Z,A,P,options);
% -------------- Initialize QW -------------- %
QW = cell(1,V);
for v = 1:V
    row_norm = sqrt(sum(W{v} .* W{v}, 2) + eps);
    QW{v} = 1 ./ (2 * row_norm);
end
alpha = updateAlpha(Xhat,W,Z,QW,options,observed);
% -------------- Optimization -------------- %
max_iter = 20;
obj_hist = zeros(max_iter, 1);
for iter = 1:max_iter
    Z = updateZ(Xhat,W,alpha,A,M,P,options,observed);
    A = updateA(Z,M,P);
    P = updateP(Z,A,M);
    M = updateM(Z,A,P,options);
    E = updateE(W,Z,G);
    for v = 1:V
       Xhat{v} = X{v}+E{v}*G{v};
    end
    alpha = updateAlpha(Xhat,W,Z,QW,options,observed);
    obj_hist(iter) = rgtal_objective(Xhat,W,Z,QW,alpha,A,M,P,options,observed);
    if iter > 1 && abs(obj_hist(iter)-obj_hist(iter-1))<1e-3
        break
    end
end
objval = obj_hist(iter);
end

function [W,Xhat,objval] = rgtal_fixed_selector_grams(X,c,options,W)
% Algebraically equivalent fixed-W updates using c-by-c/c-by-n statistics.
% Missing columns equal W*Z after each completion update, so their
% reconstruction residual is zero. This avoids d-by-n work every iteration.
V=numel(X); n=size(X{1},2);
observed=cell(1,V); B=cell(1,V); C=cell(1,V); QW=cell(1,V);
Xhat=cell(1,V); data_norm=zeros(1,V); sparse_penalty=zeros(1,V);
prox_penalty=zeros(1,V);
for v=1:V
    observed{v}=~isnan(X{v}(1,:));
    Xv=X{v}; Xv(:,~observed{v})=0; X{v}=Xv;
    Xhat{v}=Xv;
    if any(observed{v})
        Xhat{v}(:,~observed{v})=repmat( ...
            sum(Xv,2)/sum(observed{v}),1,sum(~observed{v}));
    end
    B{v}=W{v}'*Xv;
    C{v}=W{v}'*W{v};
    data_norm(v)=sum(Xv.^2,'all');
    QW{v}=1./(2*sqrt(sum(W{v}.^2,2)+eps));
    sparse_penalty(v)=trace((W{v}.*QW{v})'*W{v});
    if get_fedprox_mu(options)>0 && isfield(options,'fedprox_center') && ...
            numel(options.fedprox_center)==V && ...
            isequal(size(W{v}),size(options.fedprox_center{v}))
        prox_penalty(v)=norm(W{v}-options.fedprox_center{v},'fro')^2;
    end
end
Z=zeros(options.k,n,V);
for v=1:V
    [zr,~]=qr(rand(n,options.k),0); Z(:,:,v)=zr(:,1:options.k)';
end
A=rand(options.k,options.r); P=rand(V,options.r);
M=updateM(Z,A,P,options);
alpha=updateAlpha(Xhat,W,Z,QW,options,observed);
previous=Inf;
for iter=1:20
    mask_aware=get_option(options,'mask_aware_observed',false);
    for v=1:V
        product=B{v};
        if mask_aware
            product(:,~observed{v})=0;
        elseif iter==1
            product(:,~observed{v})=W{v}'*Xhat{v}(:,~observed{v});
        else
            product(:,~observed{v})=C{v}*Z(:,~observed{v},v);
        end
        Q=(alpha(v)^options.gamma)*product+A*diag(P(v,:))*M';
        [U,~,S]=svd(Q,'econ'); Z(:,:,v)=U*S';
    end
    A=updateA(Z,M,P); P=updateP(Z,A,M); M=updateM(Z,A,P,options);
    errors=zeros(1,V);
    for v=1:V
        Zobs=Z(:,observed{v},v);
        residual=data_norm(v)-2*sum(Zobs.*B{v}(:,observed{v}),'all') + ...
            sum(Zobs.*(C{v}*Zobs),'all');
        errors(v)=max(0,residual)+options.lambda*sparse_penalty(v);
    end
    alpha=errors.^(1/(1-options.gamma))/sum(errors.^(1/(1-options.gamma)));
    if isfield(options,'view_weights') && numel(options.view_weights)==V
        vw=max(double(options.view_weights(:))',0); vw=vw/(sum(vw)+eps);
        alpha=alpha.*vw; alpha=alpha/(sum(alpha)+eps);
    end
    objval=sum((alpha.^options.gamma).*errors)+options.eta*norm(M,'fro')^2 + ...
        .5*get_fedprox_mu(options)*sum((alpha.^options.gamma).*prox_penalty);
    for v=1:V
        objval=objval+norm(Z(:,:,v)-A*diag(P(v,:))*M','fro')^2;
    end
    if iter>1 && abs(objval-previous)<1e-3, break; end
    previous=objval;
end
if isfield(options,'fixed_selector_objective_only') && options.fixed_selector_objective_only
    Xhat={};
else
    for v=1:V
        Xhat{v}=X{v};
        Xhat{v}(:,~observed{v})=W{v}*Z(:,~observed{v},v);
    end
end
end

function [W,Xhat,objval,trace] = rgtal_optimize(X,c,options)
% RGTAL alternating updates for W, Z, A, M, P, E and alpha.
V = size(X,2);
[~,n] = size(X{1});
if ~isfield(options,'r') || isempty(options.r), options.r = c; end
if ~isfield(options,'k') || isempty(options.k), options.k = c; end

G = cell(1,V); E = cell(1,V); Xhat = cell(1,V);
observed=cell(1,V);
mask_aware=get_option(options,'mask_aware_observed',false);
for v = 1:V
    idx_v = find(isnan(X{v}(1,:)));
    observed{v}=true(1,n); observed{v}(idx_v)=false;
    ns = numel(idx_v);
    G{v} = sparse(1:ns, idx_v, 1, ns, n);
    dv = size(X{v}, 1);
    Xv = X{v};
    Xv(:, idx_v) = 0;
    if n - ns > 0
        E{v} = (sum(Xv, 2) / (n - ns)) * ones(1, ns);
    else
        E{v} = zeros(dv, ns);
    end
    X{v} = Xv;
    Xhat{v} = Xv + E{v} * G{v};
end

W = cell(1,V);
Z = zeros(options.k,n,V);
QW = cell(1,V);
for v = 1:V
    tmp = rand(n,options.k);
    [zr,~] = qr(tmp,0);
    zr = zr(:,1:options.k);
    Z(:,:,v) = zr';
    if mask_aware && any(observed{v})
        W{v} = Xhat{v}(:,observed{v})*(Z(:,observed{v},v)');
    else
        W{v} = Xhat{v}*(Z(:,:,v)');
    end
    QW{v} = ones(size(W{v}));
end

A = rand(options.k,options.r);
P = rand(V,options.r);
M = updateM(Z,A,P,options);
alpha = updateAlpha(Xhat,W,Z,QW,options,observed);

max_iter = 30;
if isfield(options, 'rgtal_max_iterations') && ~isempty(options.rgtal_max_iterations)
    max_iter = options.rgtal_max_iterations;
end
assert(isscalar(max_iter) && max_iter >= 1 && max_iter == floor(max_iter), ...
    'rgtal_max_iterations must be a positive integer.');
early_stopping = true;
if isfield(options, 'rgtal_early_stopping') && ~isempty(options.rgtal_early_stopping)
    early_stopping = logical(options.rgtal_early_stopping);
end
record_trace = nargout > 3;
trace = struct();
if record_trace
    trace.objective = nan(max_iter + 1, 1);
    trace.objective(1) = rgtal_objective(Xhat,W,Z,QW,alpha,A,M,P,options,observed);
end
obj_hist = zeros(max_iter,1);
for iter = 1:max_iter
    [W,QW] = updateW(Xhat,Z,QW,options,observed);
    Z = updateZ(Xhat,W,alpha,A,M,P,options,observed);
    A = updateA(Z,M,P);
    P = updateP(Z,A,M);
    M = updateM(Z,A,P,options);
    E = updateE(W,Z,G);
    for v = 1:V
       Xhat{v} = X{v}+E{v}*G{v};
    end
    alpha = updateAlpha(Xhat,W,Z,QW,options,observed);
    obj_hist(iter) = rgtal_objective(Xhat,W,Z,QW,alpha,A,M,P,options,observed);
    if record_trace
        trace.objective(iter + 1) = obj_hist(iter);
    end
    if early_stopping && iter > 1 && abs(obj_hist(iter)-obj_hist(iter-1)) < 1e-3
        break
    end
end
objval = obj_hist(iter);
if record_trace
    trace.iterations = (0:iter)';
    trace.objective = trace.objective(1:iter + 1);
    trace.early_stopping_enabled = early_stopping;
end
end

function f=rgtal_objective(Xhat,W,Z,QW,alpha,A,M,P,options,observed)
%% Input 
% Xhat: The imputed multi-view dataset
% W: Feature selection matrix 
% Z: Tensor formed by stacking low-dimensional representations of views.
% QW: Auxiliary matrix for computing the 2,1 norm of matrix W
% alpha: Adaptive weights for each views 
% A: Consensus anchor matrix
% M: Concensus anchor graph
% P: Anchor preference weight matrix;
% options: Parameters 
%% output
% f: objective function value

V=size(Xhat,2); % Number of views.

f1=0;
f2=0;
f3 = 0;
mask_aware=get_option(options,'mask_aware_observed',false);
for v = 1:V
    Xv_hat = Xhat{v};
    residual = Xv_hat-W{v}*Z(:,:,v);
    if mask_aware && nargin>=10 && ~isempty(observed)
        residual = residual(:,observed{v});
    end
    f1 = f1 + (alpha(v)^options.gamma)*(norm(residual,'fro')^2);  % reconstruction loss
    f2 = f2+(alpha(v)^options.gamma)*trace((W{v}.*QW{v})'*W{v}); % Sparsity regularization term.
    f3 = f3 + norm(Z(:,:,v)-A*diag(P(v,:))*M','fro')^2; % Low-dimensional consistency term.
end

f4 = norm(M,'fro')^2; % Anchor-graph regularization term.

f5 = 0;
if isfield(options, 'fedprox_mu') && options.fedprox_mu > 0 && ...
        isfield(options, 'fedprox_center') && numel(options.fedprox_center) == V
    for v = 1:V
        if isequal(size(W{v}), size(options.fedprox_center{v}))
            f5 = f5 + (alpha(v) ^ options.gamma) * ...
                norm(W{v} - options.fedprox_center{v}, 'fro') ^ 2;
        end
    end
end
f = f1 + options.lambda*f2 + f3 + options.eta*f4 + ...
    0.5 * get_fedprox_mu(options) * f5;
clear f1 f2 f3 f4
end

function A = updateA(Z,M,P)
%% Input 
% Z: Tensor formed by stacking low-dimensional representations of views
% M: Consensus anchor graph
% P: View preference weight matrix

%% output
% A: Consensus anchor matrix

% Obtain the mode-1 unfolding of tensor Z
tensor_size = size(Z);
Z_mode_1 = reshape(Z, tensor_size(1), []); % Mode-1 unfolding.
% Calculate P*M ,where * is khatri-rao product
PM = khatrirao(P,M); % Compute the Khatri-Rao product.
% Update A
A = Z_mode_1*PM/(PM'*PM); % Update A.
clear tensor_size Z_mode_1 PM
end

function [M] = updateM (Z,A,P,options)
%% Input 
% Z: Tensor formed by stacking low-dimensional representations of views
% A: Consensus anchor matrix
% P: View preference weight matrix
% options: Parameters 

%% output
% M: Consensus anchor graph

tensor_size = size(Z);
n = tensor_size(2);

% Obtain the horizontal mode-2 unfolding of tensor Z
Z_mode_2 = reshape(permute(Z, [2 1 3]), tensor_size(2), []); % Mode-2 unfolding.
I = eye(options.r); % r is the number of anchors.

% Calculate P*A ,where * is khatri-rao product
PA = khatrirao(P,A); % Compute the Khatri-Rao product.

Y = -1*Z_mode_2*(PA)/(options.eta*I+(PA')*PA); % Compute Y.

% Update M
v1 = ones(options.r,1); 
tmp = ((1+Y*v1)/options.r).*ones(n,options.r)-Y; % Intermediate variable for updating M.
M = max(tmp,0); % Keep the nonnegative part.
clear tensor_size Z_mode_2 PA Y tmp
end

function P = updateP(Z,A,M)
%% Input 
% Z: Tensor formed by stacking low-dimensional representations of views
% A: Consensus anchor matrix
% M: Consensus anchor graph
%% output
% P: View preference weight matrix

% Obtain the horizontal mode-3 unfolding of tensor Z
tensor_size = size(Z);
Z_mode_3 = reshape(permute(Z, [3 1 2]), tensor_size(3), []); % Mode-3 unfolding.

% Calculate P*M ,where * is khatri-rao product
MA = khatrirao(M,A); % Compute the Khatri-Rao product.
% Update P
P = Z_mode_3*MA/(MA'*MA);   % Update P.
clear Z_mode_3 tensor_size MA
end 

function [alpha] = updateAlpha(Xhat,W,Z,QW,options,observed)
%% Input 
% Xhat: The imputed multi-view dataset
% W: Feature selection matrix 
% Z: Tensor formed by stacking low-dimensional representations of views
% QW: Auxiliary matrix for computing the 2,1 norm of matrix W
% options: parameters 
%% output
% alpha: Adaptive weights of each view

V = size(Xhat,2); 
mask_aware=get_option(options,'mask_aware_observed',false);

d = zeros(1,V); 
for v = 1:V
   residual=Xhat{v}-W{v}*Z(:,:,v);
   if mask_aware && nargin>=6 && ~isempty(observed)
       residual=residual(:,observed{v});
   end
   d(v) = norm(residual,'fro')^2 + ...
        options.lambda*trace((W{v}.*QW{v})'*W{v});   % Compute the loss for each view.
end

% Update alpha
alpha = d.^(1/(1-options.gamma))/sum(d.^(1/(1-options.gamma))); % Update the weights.
if isfield(options, 'view_weights') && ~isempty(options.view_weights)
    vw = double(options.view_weights(:))';
    if numel(vw) == V
        vw = max(vw, 0);
        vw = vw / (sum(vw) + eps);
        alpha = alpha .* vw;
        alpha = alpha / (sum(alpha) + eps);
    end
end
end

function [E] = updateE(W,Z,G)
%% Input 
% W: Feature selection matrix 
% Z: Tensor formed by stacking low-dimensional representations of views
% G: Missing samples indicator matrix for each view 
% options: Parameters 
%% output
% E: Adaptive imputation matrix for missing samples in each view

V = size(W,2); 

% Update E
E = cell(1,V);
for v =1:V
    E{v} = full(W{v}*(Z(:,:,v)*G{v}')); % Compute the completed matrix.
end

end

function [W,QW] = updateW(Xhat,Z,QW,options,observed)
%% Input 
% Xhat: The imputed multi-view dataset
% Z: Tensor formed by stacking low-dimensional representations of views
% QW: Auxiliary matrix for computing the 2,1 norm of matrix W
% options: Parameters

%% output
% W: Feature selection matrix
% QW: Auxiliary matrix for computing the 2,1 norm of matrix W

V = size(Xhat,2);
W = cell(1,V);
mask_aware=get_option(options,'mask_aware_observed',false);

for v = 1:V
    Xv_hat = Xhat{v};
    if mask_aware && nargin>=5 && ~isempty(observed)
        obs=observed{v};
        Zobs=Z(:,obs,v);
        numerator=Xv_hat(:,obs)*Zobs';
        gram=Zobs*Zobs';
        % The objective uses ||X-WZ||^2 + lambda*tr(W'*Q*W)
        % + (fedprox_mu/2)*||W-center||^2. Dividing its gradient
        % by two gives fedprox_mu/2 in the normal equations.
        mu=0.5*get_fedprox_mu(options);
        center=[];
        if mu>0 && isfield(options,'fedprox_center') && ...
                numel(options.fedprox_center)==V && ...
                isequal(size(options.fedprox_center{v}),size(numerator))
            center=options.fedprox_center{v};
        else
            mu=0;
        end
        Wv=zeros(size(numerator)); eye_k=eye(size(gram));
        for row=1:size(numerator,1)
            ridge=options.lambda*QW{v}(row,1)+mu;
            rhs=numerator(row,:)';
            if mu>0, rhs=rhs+mu*center(row,:)'; end
            Wv(row,:)=(gram+ridge*eye_k)\rhs;
        end
        W{v}=Wv;
    else
        numerator = Xv_hat * Z(:,:,v)';
        denominator = 1 + options.lambda * QW{v};
        mu=0.5*get_fedprox_mu(options);
        if mu > 0 && ...
            isfield(options, 'fedprox_center') && numel(options.fedprox_center) == V && ...
            isequal(size(options.fedprox_center{v}), size(numerator))
            numerator = numerator + mu * options.fedprox_center{v};
            denominator = denominator + mu;
        end
        W{v} = numerator ./ denominator; % W update with optional FedProx anchor
    end
    QW{v} = 1./(2*sqrt(sum(W{v}.*W{v},2)+eps)); % Update QW.
end

end

function [Z] = updateZ(Xhat,W,alpha,A,M,P,options,observed)
%% Input 
% Xhat: The imputed multi-view dataset
% W: Feature selection matrix 
% alpha: Adaptive weights for each view
% A: Consensus anchor matrix
% M: Consensus anchor graph
% P: Anchor preference weight matrix;
% options: Parameters 
%% output
% Z: Tensor formed by stacking low-dimensional representations of views.

V = size(Xhat,2); % Number of views.
mask_aware=get_option(options,'mask_aware_observed',false);

[~,n] = size(Xhat{1}); % Number of samples.

% Update Z
Z = zeros(options.k,n,V); % Initialize Z.
for v = 1:V
    Cv = A*diag(P(v,:))*M'; % Compute the consistency term.
    Xv=Xhat{v};
    if mask_aware && nargin>=8 && ~isempty(observed)
        Xv(:,~observed{v})=0;
    end
    Qv = (alpha(v)^options.gamma)*W{v}'*Xv+Cv;% Intermediate variable.
    [Uv, ~, Vv] = svd(Qv,'econ'); % Singular value decomposition.
    Z(:,:,v) = Uv*Vv'; % Update Z.
end
end

function value=get_option(options,name,fallback)
if isfield(options,name) && ~isempty(options.(name))
    value=options.(name);
else
    value=fallback;
end

end



