# ECE1512 Project A - PyTorch MIL on WSI features

The project uses Attention-Based Multiple Instance Learning (ABMIL) for whole-slide histopathology classification.

## 1. Download the pre-extracted features

The pre-extracted histopathology features are hosted on Zenodo:

https://zenodo.org/records/23130543


Download the HDF5 feature files and place them in the designated `feature` folder.

```text
feature/
  C16/patch_feats_pretrain_medical_ssl.h5
  C17/patch_feats_pretrain_medical_ssl.h5
  BRACS/patch_feats_pretrain_medical_ssl.h5
```

If you store the HDF5 feature files elsewhere, edit only `data_dir` in the selected YAML file.

## 2. Coding

Open:

```text
ECE1512_ProjectA_Student.ipynb
```

The notebook is incomplete. Search for **TODO** and complete the requested lines.

## 3. What is provided

- `dataloader.py` - HDF5 reading, dataset-specific splitting, few-shot selection, and padded MIL collation.
- `utils.py` - configuration loading, seeding, feature-dimension checks, and AUROC/F1 metric helpers.
- `config/` - templates for CAMELYON16, CAMELYON17, and BRACS.
- `dataset_csv/` - metadata required by the CAMELYON17 and BRACS split logic.
- `ECE1512_ProjectA_Student.ipynb` - guided PyTorch implementation, baseline experiment, and ablation workspace.

The completed ABMIL model and training loop are deliberately **not** included in the student starter package.

## 4. Expected HDF5 layout

Each top-level HDF5 key is one slide:

```text
slide_id/
  feat    [num_patches, feature_dim]
  coords  [num_patches, 2]   # optional for training
attrs:
  label   integer class label
```

Different slides contain different numbers of patches. `mil_collate_fn` pads a batch and returns a boolean `mask` that marks valid patches.

## 5. Colab/local setup

Install dependencies with:

```bash
pip install -r requirements.txt
```

In Colab, place the project folder in Google Drive or upload it to the runtime, set `PROJECT_DIR` in the notebook, and select `C16`, `C17`, or `BRACS`.

For an initial debugging run, use only a few epochs. Once your pipeline passes the sanity checks, run the full baseline and keep the same split, seed, and evaluation protocol for your ablations.

## 6. Baseline

### Running it

- Single run: set `DATASET` in Step 0 and run Steps 0-8.
- All datasets (C16, C17, BRACS) × seeds (42, 0, 1): run Steps 0-5, then Step 8b.

The split is fixed by `seed: 42` in each YAML, so the run seed only changes weight init and shuffling. Step 8b skips runs that already have a JSON in `results/`, so with the provided results it only rebuilds the tables and figures. Set `RERUN = True` to retrain.

The C16 WSI overlays need `test_067.tif`, `test_068.tif` and `test_068.xml` from [Hugging Face](https://huggingface.co/datasets/LeonYe/ECE1512-WSI-Feature) in `slide/`.

### Results

Everything is already in `results/`:

```text
results/
  {DATASET}_seed{SEED}.json                  # history, best epoch, test metrics
  {DATASET}_seed{SEED}_test_predictions.csv  # per-slide test predictions
  test_metrics_all_runs.csv
  test_metrics_mean_std.csv
  ckpt/{DATASET}_seed{SEED}.pt
  figures/
    loss_{DATASET}.png, loss_{DATASET}_seed{SEED}.png
    attention_test_067_seed{SEED}.png, attention_test_068_seed{SEED}.png  # C16, on the WSI
    attention_map_{DATASET}.png                                           # on patch coordinates
```

Test metrics, mean ± std over 3 seeds (macro AUROC and F1):

| Dataset | Classes | Test slides | AUROC | Accuracy (%) | Macro-F1 |
|---|---|---|---|---|---|
| CAMELYON16 | 2 | 129 | 0.925 ± 0.021 | 89.4 ± 3.2 | 0.881 ± 0.038 |
| CAMELYON17 | 4 | 200 | 0.848 ± 0.035 | 78.0 ± 3.9 | 0.515 ± 0.050 |
| BRACS | 7 | 86 | 0.799 ± 0.012 | 46.9 ± 1.8 | 0.374 ± 0.012 |

For C17 and BRACS there are no WSIs, so attention is plotted at the patch coordinates for one correct and one misclassified test slide.

## 7. Extension: stochastic bag subsampling (Step 9)

Each epoch, every training slide keeps a random `ceil(rho * N)` of its `N` patches. Validation and test use full bags. Everything else matches the baseline, and `rho = 1` is the baseline.

### Running it

Run Steps 0-5, the Step 8b helpers cell (not the training loop), then the Step 9 cells in order. Step 9:

1. Checks that `rho = 1` reproduces the first 3 epochs of `results/C16_seed42.json`.
2. Trains `rho` in {0.25, 0.5, 0.75} on each dataset with seeds 42, 0 and 1 (27 runs).
3. Picks `rho` per dataset by mean validation AUC + F1 (ties go to the lower validation loss), then compares it with the baseline on test. The other `rho` values are shown as a sensitivity analysis.

Settings are in the Step 9 config cell (`RHOS`, `EXT_SEEDS`, `EXT_NUM_EPOCHS`, `EXT_RERUN`, `EXT_DIR`). `TRACK_FULL_TRAIN_LOSS` adds a full-bag pass over the training set each epoch, so the training loss can be compared with the baseline. Finished runs are skipped.

### Results

```text
results/ext_subsample/
  {DATASET}_rho{RHO}_seed{SEED}.json, ..._test_predictions.csv
  ckpt/, figures/, sanity/
  rho_selection_validation.csv, chosen_rho.json  # rho chosen on validation
  main_comparison.csv                            # baseline vs chosen rho on test
  sensitivity_all_runs.csv, sensitivity_rho.png  # every rho on test
  overfitting_summary.csv, loss_curves_baseline_vs_rho.png
  per_class_f1_recall.csv, per_class_f1_vs_rho.png
```
