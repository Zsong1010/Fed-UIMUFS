function Summary = RunFedUIMUFS_SGHBO(dataFile, config)
%RUNFEDUIMUFS_SGHBO Run one multi-view MAT input with the paper method.
% Use Main_FedUIMUFS as the user-facing entry point.
% MAT file must contain X or X_missing (cell array, one feature-by-sample
% matrix per view) and Y. Labels are used only for final evaluation.
if nargin < 1 || isempty(dataFile)
    error('Provide a multi-view MAT file containing X or X_missing and Y.');
end
if nargin < 2 || isempty(config), config=struct(); end
config=with_default(config,'DatasetName',erase(string(get_filename(dataFile)),'.mat'));
config=with_default(config,'Seeds',[2101 2102 2103]);
config=with_default(config,'Clients',5);
config=with_default(config,'FeatureRatio',0.4);
config=with_default(config,'MaxRounds',10);
config=with_default(config,'RequireUnbalanced',true);
config=with_default(config,'RequireIncomplete',true);
config=with_default(config,'ResultsRoot','');
config=with_default(config,'SGHBOParticles',15);
config=with_default(config,'SGHBOIterations',10);
config=with_default(config,'ImputationStandardization','feature_zscore');
config=with_default(config,'MaskAwareObserved',false);
config=with_default(config,'FederatedEarlyStopping',true);
config=with_default(config,'FederatedTolerance',1e-4);
config=with_default(config,'Gamma',6);
config=with_default(config,'ParallelSeeds',false);
config=with_default(config,'ParallelSeedWorkers',0);
config=with_default(config,'ParallelClients',false);
config=with_default(config,'ParallelClientWorkers',0);
assert(any(strcmpi(string(config.ImputationStandardization),["feature_zscore","none"])), ...
    'ImputationStandardization must be feature_zscore or none.');
assert(isfile(dataFile),'Input MAT file not found: %s',dataFile);
assert(~isempty(config.Seeds) && numel(unique(config.Seeds))==numel(config.Seeds), ...
    'Seeds must be a nonempty vector without duplicates.');
assert(config.Clients>=2 && config.Clients==floor(config.Clients),'Clients must be an integer >=2.');
assert(config.FeatureRatio>0 && config.FeatureRatio<=1,'FeatureRatio must be in (0,1].');
assert(config.MaxRounds>=1 && config.MaxRounds==floor(config.MaxRounds),'MaxRounds must be a positive integer.');
assert(isscalar(config.Gamma) && isfinite(config.Gamma) && config.Gamma>1, ...
    'Gamma must be a finite scalar greater than one.');
assert(isscalar(config.ParallelSeedWorkers) && config.ParallelSeedWorkers>=0 && ...
    config.ParallelSeedWorkers==floor(config.ParallelSeedWorkers), ...
    'ParallelSeedWorkers must be a nonnegative integer.');
assert(isscalar(config.ParallelClientWorkers) && config.ParallelClientWorkers>=0 && ...
    config.ParallelClientWorkers==floor(config.ParallelClientWorkers), ...
    'ParallelClientWorkers must be a nonnegative integer.');
assert(~(config.ParallelSeeds && config.ParallelClients), ...
    'Enable either seed-level or client-level parallelism, not both.');
if config.ParallelSeeds || config.ParallelClients
    assert(license('test','Distrib_Computing_Toolbox'), ...
        'Parallel execution requires MATLAB Parallel Computing Toolbox.');
end

root=fileparts(fileparts(mfilename('fullpath')));
addpath(root,fullfile(root,'ClientOptimization'),fullfile(root,'CalcMeasures'));
assert(strcmp(which('FederatedFeatureSelection'),fullfile(root,'FederatedFeatureSelection.m')));
assert(strcmp(which('ClientCoupledOptimization'),fullfile(root,'ClientOptimization','ClientCoupledOptimization.m')));
D=load(dataFile);
if isfield(D,'X'), X=D.X; elseif isfield(D,'X_missing'), X=D.X_missing;
else, error('MAT file must contain X or X_missing as a cell array.'); end
assert(iscell(X) && ~isempty(X),'X must be a nonempty cell array.');
assert(isfield(D,'Y'),'MAT file must contain Y for final evaluation.');
[~,~,Y]=unique(D.Y(:)); Y=double(Y);
N=numel(Y); V=numel(X); c=numel(unique(Y));
missing=cell(1,V); rates=zeros(1,V); dims=zeros(1,V);
for v=1:V
    if size(X{v},2)~=N && size(X{v},1)==N, X{v}=X{v}'; end
    assert(size(X{v},2)==N,'Sample count mismatch in view %d.',v);
    X{v}=full(double(X{v}));
    missing{v}=all(isnan(X{v}),1);
    rates(v)=mean(missing{v}); dims(v)=size(X{v},1);
    assert(all(isfinite(X{v}(:,~missing{v})),'all'), ...
        'Observed entries must be finite and missingness must be view-wise (view %d).',v);
end
meanRate=mean(rates);
assert(any(rates>0),'The input dataset is complete; this method entry expects incomplete views.');
if config.RequireUnbalanced
    assert(max(rates)-min(rates)>0.05, ...
        'Input is incomplete but not sufficiently unbalanced; view missing rates=%s.',mat2str(rates,5));
end
[Ximputed,weights,imputationStats]=ImputeUnbalancedViews(X, ...
    struct('standardization',char(config.ImputationStandardization)));
for v=1:V
    A=Ximputed{v}; A(~isfinite(A))=0;
    A=A./(sqrt(sum(A.^2,1))+eps);
    A(:,missing{v})=NaN; X{v}=A;
end
dataset=char(regexprep(string(config.DatasetName),'[^A-Za-z0-9_-]','_'));
if strlength(string(config.ResultsRoot))>0
    parent=char(config.ResultsRoot);
else
    parent=fullfile(root,'results');
end
stamp=char(datetime('now','Format','yyyyMMdd_HHmmss'));
out=fullfile(parent,[dataset '_SGHBO_FR' strrep(num2str(config.FeatureRatio,'%.2f'),'.','') ...
    '_MR' strrep(num2str(meanRate*100,'%.1f'),'.','') '_' stamp]);
mkdir(out);
protocol=struct('dataset',dataset,'source_data',char(dataFile),'source_sha256',hash_file(dataFile), ...
    'instances',N,'classes',c,'views',V,'view_dimensions',dims, ...
    'view_missing_rates',rates,'mean_missing_rate',meanRate, ...
    'clients',config.Clients,'seeds',config.Seeds,'feature_ratio',config.FeatureRatio, ...
    'maximum_rounds',config.MaxRounds,'method','RGTAL-SGHBO + secure RAFA + corrected FedProx', ...
    'preprocessing',char("ImputeUnbalancedViews("+string(config.ImputationStandardization)+") + column L2; input missing masks preserved"), ...
    'mask_aware_observed',logical(config.MaskAwareObserved), ...
    'federated_early_stopping',logical(config.FederatedEarlyStopping), ...
    'federated_tolerance',config.FederatedTolerance, ...
    'gamma',config.Gamma, ...
    'parallel_seeds',logical(config.ParallelSeeds), ...
    'parallel_seed_workers',config.ParallelSeedWorkers, ...
    'parallel_clients',logical(config.ParallelClients), ...
    'parallel_client_workers',config.ParallelClientWorkers, ...
    'communication_accounting','float64 numeric payload only; includes secure-RAFA upload vector and K selector broadcasts; excludes headers, keys, and network framing', ...
    'evaluation','spectral clustering on selected features; 30 K-means restarts', ...
    'require_unbalanced',logical(config.RequireUnbalanced),'require_incomplete',logical(config.RequireIncomplete), ...
    'source_code_sha256',struct( ...
        'FederatedFeatureSelection',hash_file(fullfile(root,'FederatedFeatureSelection.m')), ...
        'ClientCoupledOptimization',hash_file(fullfile(root,'ClientOptimization','ClientCoupledOptimization.m')), ...
        'ImputeUnbalancedViews',hash_file(fullfile(root,'ImputeUnbalancedViews.m')), ...
        'RunFedUIMUFS_SGHBO',hash_file([mfilename('fullpath') '.m'])), ...
    'created',char(datetime('now')));
fid=fopen(fullfile(out,'protocol.json'),'w'); assert(fid>=0);
fprintf(fid,'%s',jsonencode(protocol,'PrettyPrint',true)); fclose(fid);
save(fullfile(out,'design.mat'),'protocol','missing','rates','weights','imputationStats','Y','config','-v7.3');
fprintf('%s: N=%d classes=%d V=%d dimensions=%s viewMR=%s meanMR=%.6f\n', ...
    dataset,N,c,V,mat2str(dims),mat2str(rates,5),meanRate);
rows=table();
seedList=config.Seeds(:)';
if config.ParallelSeeds || config.ParallelClients
    pool=gcp('nocreate');
    if config.ParallelSeeds
        requestedWorkers=config.ParallelSeedWorkers;
    else
        requestedWorkers=config.ParallelClientWorkers;
    end
    if isempty(pool)
        if requestedWorkers>0
            parpool('Processes',requestedWorkers);
        else
            parpool('Processes');
        end
    elseif requestedWorkers>0 && pool.NumWorkers~=requestedWorkers
        error('RunFedUIMUFS_SGHBO:WorkerCountMismatch', ...
            'The existing parallel pool has %d workers, but %d were requested.', ...
            pool.NumWorkers,requestedWorkers);
    end
end
if config.ParallelSeeds
    rowCells=cell(numel(seedList),1);
    parfor seedIndex=1:numel(seedList)
        seed=seedList(seedIndex);
        rowCells{seedIndex}=run_seed(seed,out,dataset,X,Y,V,c,N,config.Clients,config.MaxRounds, ...
            config.FeatureRatio,weights,missing,dims,config.SGHBOParticles,config.SGHBOIterations, ...
            char(config.ImputationStandardization),logical(config.MaskAwareObserved), ...
            logical(config.FederatedEarlyStopping),config.FederatedTolerance,config.Gamma, ...
            logical(config.ParallelClients));
    end
    rows=vertcat(rowCells{:});
else
for seed=seedList
    Row=run_seed(seed,out,dataset,X,Y,V,c,N,config.Clients,config.MaxRounds, ...
        config.FeatureRatio,weights,missing,dims,config.SGHBOParticles,config.SGHBOIterations, ...
        char(config.ImputationStandardization),logical(config.MaskAwareObserved), ...
        logical(config.FederatedEarlyStopping),config.FederatedTolerance,config.Gamma, ...
        logical(config.ParallelClients));
    rows=[rows;Row]; %#ok<AGROW>
end
end
writetable(rows,fullfile(out,'all_runs.csv'));
metricCols={'ACC_Percent','NMI_Percent','Purity_Percent','PairwiseF1_Percent','ARI_Percent'};
metricNames=string({'ACC';'NMI';'Purity';'Pairwise F1';'ARI'});
Summary=table(repmat(string(dataset),5,1),metricNames,mean(rows{:,metricCols},1)', ...
    std(rows{:,metricCols},0,1)',max(rows{:,metricCols},[],1)',repmat(100*meanRate,5,1), ...
    repmat(string(mat2str(100*rates,4)),5,1), ...
    'VariableNames',{'Dataset','Metric','Mean_Percent','SD_Percent','Best_Percent', ...
    'MeanMissingRate_Percent','ViewMissingRates_Percent'});
writetable(Summary,fullfile(out,'summary_mean_sd.csv'));
efficiency=table(string(dataset),mean(rows.Rounds),std(rows.Rounds,0), ...
    mean(rows.StoppedByTolerance),mean(rows.TotalPayloadBytes),std(rows.TotalPayloadBytes,0), ...
    mean(rows.TotalPayloadMiB),mean(rows.UplinkPayloadBytesPerRound), ...
    mean(rows.DownlinkPayloadBytesPerRound), ...
    'VariableNames',{'Dataset','MeanRounds','SDRounds','ConvergedFraction', ...
    'MeanTotalPayloadBytes','SDTotalPayloadBytes','MeanTotalPayloadMiB', ...
    'MeanUplinkPayloadBytesPerRound','MeanDownlinkPayloadBytesPerRound'});
writetable(efficiency,fullfile(out,'efficiency_summary.csv'));
save(fullfile(out,'all_runs.mat'),'rows','Summary','protocol','config','-v7.3');
fprintf('\nPer-seed results:\n'); disp(rows);
fprintf('Summary across %d run(s):\n',height(rows)); disp(Summary);
fprintf('Results saved under %s\n',out);
end

function Row=run_seed(seed,out,dataset,X,Y,V,c,N,K,T,ratio,weights,missing,dims,nPop,nIter,imputationStandardization,maskAwareObserved,federatedEarlyStopping,federatedTolerance,gamma,parallelClients)
maxNumCompThreads(1);
stem=sprintf('%s_SGHBO_seed%d',dataset,seed);
options=struct('gamma',gamma,'lambda',0.001,'eta',1,'view_weights',weights, ...
    'use_sghbo',true,'use_legacy_hbo',false,'use_rafa',true,'secure_aggregation',true, ...
    'fast_fixed_selector',true,'fedprox_mu',0.01,'federated_tolerance',federatedTolerance, ...
    'federated_early_stopping',federatedEarlyStopping, ...
    'parallel_clients',parallelClients,'parallel_seed',seed, ...
    'federated_record_local_trace',true,'rafa_zeta',0.1, ...
    'rafa_nu_n',1,'rafa_nu_q',1,'rafa_nu_R',1,'rafa_nu_v',1,'rafa_epsilon',1e-8, ...
    'rafa_stability_seed',seed,'sghbo_fitness_seed',seed,'sghbo_particles',nPop, ...
    'sghbo_iterations',nIter,'sghbo_eta',0.2,'sghbo_lambda_sg',0.05, ...
    'sghbo_con_weight',0.5,'sghbo_lambda_w',0.25,'sghbo_acm_damping',0.25, ...
    'imputation_standardization',imputationStandardization, ...
    'mask_aware_observed',maskAwareObserved, ...
    'sghbo_beta_min',0.1,'sghbo_beta_max',0.9,'sghbo_theta',0.5,'sghbo_nu',1,'sghbo_xi',1);
rng(seed,'twister'); partition=FederatedPartition(N,K); timer=tic;
console=evalc('[W,objective,Xhat,history]=FederatedFeatureSelection(X,zeros(N,1),c,V,K,partition,options,T);');
seconds=toc(timer);
[values,evaluation]=evaluate_features(Xhat,Y,W,c,weights,ratio,seed+70000);
calls=0;
for k=1:K
    for t=1:history.stop_round
        trace=history.local_trace{k,t};
        assert(isfield(trace,'sghbo') && all(isfinite(trace.sghbo.best_fitness)) && ...
            all(trace.sghbo.best_fitness<1e12),'Missing/invalid SGHBO trace.');
        calls=calls+1;
    end
end
assert(calls==K*history.stop_round,'SGHBO did not execute for every client-round.');
Row=table(string(dataset),"RGTAL-SGHBO + secure RAFA (FedProx corrected)",seed,K,ratio, ...
    100*mean(cellfun(@mean,missing)),100*values(1),100*values(2),100*values(3), ...
    100*values(4),100*values(5),history.stop_round,history.converged,calls, ...
    evaluation.selected_count,sum(dims),seconds,objective, ...
    'VariableNames',{'Dataset','Method','Seed','Clients','FeatureRatio','MeanMissingRate_Percent', ...
    'ACC_Percent','NMI_Percent','Purity_Percent','PairwiseF1_Percent','ARI_Percent', ...
    'Rounds','StoppedByTolerance','SGHBOCalls','SelectedFeatures','OriginalFeatures', ...
    'TrainingSeconds','FinalMeanObjective'});
selector_entries=sum(cellfun(@numel,W));
upload_scalars_per_client_round=2*selector_entries+2*V+1;
downlink_scalars_per_client_round=selector_entries;
uplink_bytes_per_round=8*K*upload_scalars_per_client_round;
downlink_bytes_per_round=8*K*downlink_scalars_per_client_round;
total_payload_bytes=(uplink_bytes_per_round+downlink_bytes_per_round)*history.stop_round;
Row.SelectorEntriesPerClient=selector_entries;
Row.UplinkPayloadBytesPerRound=uplink_bytes_per_round;
Row.DownlinkPayloadBytesPerRound=downlink_bytes_per_round;
Row.TotalPayloadBytes=total_payload_bytes;
Row.TotalPayloadMiB=total_payload_bytes/(1024^2);
localRates=zeros(K,V);
for k=1:K, for v=1:V, localRates(k,v)=mean(missing{v}(partition{k})); end; end
save(fullfile(out,[stem '_result.mat']),'Row','W','Xhat','history','options','partition', ...
    'missing','Y','evaluation','localRates','console','-v7.3');
fid=fopen(fullfile(out,[stem '_log.txt']),'w'); assert(fid>=0);
fprintf(fid,'%s',console); fclose(fid);
writetable(Row,fullfile(out,[stem '_metrics.csv']));
fprintf('%s seed %d: ACC %.3f NMI %.3f Purity %.3f PairwiseF1 %.3f ARI %.3f; rounds=%d SGHBO=%d\n', ...
    dataset,seed,100*values,history.stop_round,calls);
end

function [values,details]=evaluate_features(Xhat,Y,W,c,weights,ratio,seed)
weights=max(double(weights(:))',0); weights=weights/(sum(weights)+eps);
scores=cell(1,numel(W));
for v=1:numel(W), scores{v}=sum(W{v}.^2,2)*weights(v); end
[~,order]=sort(vertcat(scores{:}),'descend');
keep=max(1,min(numel(order),round(ratio*numel(order))));
F=vertcat(Xhat{:}); F=F(order(1:keep),:)'; F=F./(sqrt(sum(F.^2,2))+eps);
D=max(0,1-F*F'); sigma=median(D(D>0))+eps;
A=exp(-(D.^2)/(2*sigma^2)); n=numel(Y); A(1:n+1:end)=0;
degree=sum(A,2); L=diag(1./sqrt(degree+eps))*A*diag(1./sqrt(degree+eps)); L=(L+L')/2;
rng(seed,'twister'); [U,~]=eigs(L,c,'largestreal'); U=real(U);
U=U./(sqrt(sum(U.^2,2))+eps);
yhat=KMeansSimple(U,c,30,400,seed+1); mapped=bestMap(Y,yhat);
values=[mean(mapped(:)==Y(:)),MutualInfo(Y,mapped),Purity(Y,mapped),Fscore(Y,mapped),ARI(Y,mapped)];
assert(all(isfinite(values)) && values(3)+1e-12>=values(1));
details=struct('selected_count',keep,'selected_indices',order(1:keep), ...
    'predictions',yhat,'evaluation_seed',seed,'kmeans_restarts',30,'feature_ratio',ratio);
end

function config=with_default(config,name,value)
if ~isfield(config,name) || isempty(config.(name)), config.(name)=value; end
end

function name=get_filename(path)
[~,name,ext]=fileparts(path); name=[name ext];
end

function value=hash_file(path)
fid=fopen(path,'rb'); assert(fid>=0,'Cannot hash file: %s',path);
bytes=fread(fid,inf,'*uint8'); fclose(fid);
md=java.security.MessageDigest.getInstance('SHA-256'); md.update(bytes);
raw=typecast(md.digest(),'uint8'); value=lower(reshape(dec2hex(raw,2)',1,[]));
end


