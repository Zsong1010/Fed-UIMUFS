function Summary = Main_FedUIMUFS(dataFile, config)
root = fileparts(mfilename('fullpath'));

if nargin < 1 || isempty(dataFile)
    [fileName, folderName] = uigetfile('*.mat', 'Select a multi-view MAT file');
    if isequal(fileName, 0)
        error('No dataset was selected. Pass a MAT-file path or select a file.');
    end
    dataFile = fullfile(folderName, fileName);
end
assert(isfile(dataFile), 'Input MAT file not found: %s', dataFile);

if nargin < 2 || isempty(config)
    config = struct();
end
assert(isstruct(config), 'config must be a struct.');
[~, defaultDatasetName] = fileparts(char(dataFile));
config = set_default(config, 'DatasetName', defaultDatasetName);
config = set_default(config, 'Seeds', 2101:2130);
config = set_default(config, 'Clients', 5);
config = set_default(config, 'FeatureRatio', 0.4);
config = set_default(config, 'MaxRounds', 10);
config = set_default(config, 'RequireUnbalanced', true);
config = set_default(config, 'RequireIncomplete', true);
config = set_default(config, 'ResultsRoot', fullfile(root, 'results', 'reruns'));
config = set_default(config, 'RSHBOParticles', 15);
config = set_default(config, 'RSHBOIterations', 10);
config = set_default(config, 'ImputationStandardization', 'feature_zscore');
config = set_default(config, 'MaskAwareObserved', false);
config = set_default(config, 'FederatedEarlyStopping', true);
config = set_default(config, 'FederatedTolerance', 1e-4);
config = set_default(config, 'Gamma', 2);
config = set_default(config, 'ParallelSeeds', true);
config = set_default(config, 'ParallelSeedWorkers', 8);
config = set_default(config, 'ParallelClients', false);
config = set_default(config, 'ParallelClientWorkers', 0);

addpath(fullfile(root, 'experiments'));
Summary = RunFedUIMUFS_RSHBO(dataFile, config);
end

function config = set_default(config, fieldName, defaultValue)
if ~isfield(config, fieldName) || isempty(config.(fieldName))
    config.(fieldName) = defaultValue;
end
end
