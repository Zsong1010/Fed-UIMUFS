# Fed-UIMUFS

MATLAB implementation of the local coupled optimization (URTAL-RSHBO) method, with corrected FedProx updates and pairwise-masked RAFA aggregation. The experiment code accepts any compatible multi-view MAT file and is not tied to a particular dataset. Reference metrics use Yale data, while the raw Yale MAT files are excluded from the public source release until redistribution terms are confirmed. Bring your own authorized input data; see `DATASET_NOTICE.md`.

## Run on a dataset

From the project root in MATLAB, either select a MAT file in the file picker:

```matlab
Summary = Main_FedUIMUFS;
```

Or pass a path directly:

```matlab
Summary = Main_FedUIMUFS('dataset/my_dataset.mat');
```

`Main_FedUIMUFS.m` is the user-facing main function. To use another dataset, select its file or change the path in the call above. The function name, experiment driver, and default protocol are dataset-independent. The lower-level runner is `experiments/RunFedUIMUFS_RSHBO.m`.

## Input MAT-file format

The MAT file must contain:

- `X` or `X_missing`: a cell array with one numeric matrix per view. Each matrix is features-by-samples; the runner also detects and transposes samples-by-features matrices.
- `Y`: one class label per sample, used only for final clustering metrics.

Represent a missing view for a sample with a column containing only `NaN`. Observed values must be finite. By default, the runner requires the input to be both incomplete and unbalanced across views. Set `RequireUnbalanced=false` in the optional config to allow equal view-wise missing rates.

## Default experiment settings

The main function uses seeds 2101–2130, five simulated clients, feature ratio 0.4, at most 10 communication rounds, 15 RSHBO particles, 10 RSHBO iterations, Gamma=2, feature-wise z-scoring, early stopping at tolerance 1e-4, and eight parallel seed workers. It requires MATLAB R2025b or compatible and Parallel Computing Toolbox. New runs are written under `results/reruns/`, which is excluded by `.gitignore`.

Pass a config struct as the second argument to override settings. For example, a quick one-seed run is:

```matlab
cfg = struct('Seeds', 2101, 'ParallelSeeds', false);
Summary = Main_FedUIMUFS('dataset/my_dataset.mat', cfg);
```

## Bundled Yale reference results

The checked-in reference results correspond to Yale inputs at mean missing rates 0.1 and 0.4. The raw Yale MAT files are excluded from the public source release pending confirmation of redistribution terms. The checked-in results are the means of 30 seeds; percentages are shown below. F-score is the project's pairwise F1 metric.

| Mean missing rate | ACC | NMI | Purity | F-score | ARI |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 60.32 | 61.71 | 60.63 | 43.55 | 39.74 |
| 0.4 | 47.82 | 49.88 | 49.11 | 28.49 | 23.66 |

`results/metrics_summary.csv` contains the compact table. Each `results/Yale_MR*/` folder contains per-seed metrics, mean/standard-deviation and efficiency summaries, and the run protocol. These are reference results for the bundled examples; changing the input file makes the same driver run another compatible dataset.

## Method and scope

Client optimization and masked aggregation are simulated in one MATLAB process; this is not a physically distributed deployment or a formal privacy guarantee. The third-party `khatrirao` license is in `LICENSES/`. The root MIT license covers project-authored software only. Dataset terms are separate; see `DATASET_NOTICE.md` before redistributing the bundled Yale MAT files.
