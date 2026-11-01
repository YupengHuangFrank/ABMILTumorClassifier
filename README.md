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

## 6. What was accomplished

### ABMIL baseline implementation

All `TODO` cells in `ECE1512_ProjectA_Student.ipynb` are complete:

- **Data:** train/validation/test `DataLoader`s (only the training loader is shuffled).
- **Model:** gated-attention ABMIL (Ilse et al., 2018). Patch features are reduced to 256 dimensions (linear + ReLU), scored by a gated attention branch, softmax-normalised over the patches of each slide, pooled with `torch.bmm`, and classified by a linear layer.
- **Training:** cross-entropy loss, AdamW (`lr=1e-4`, `weight_decay=1e-5`), 30 epochs, and the standard `zero_grad -> forward -> loss -> backward -> step` loop.
- **Model selection:** the epoch with the best validation macro-AUROC + macro-F1 is kept. The test set is evaluated once, with that checkpoint.

`utils.py`, `dataloader.py` and `wsi_visualization.py` are the instructor-provided files and are unchanged.

### Multi-dataset, multi-seed runs (notebook Step 8b)

Step 8b reruns the same pipeline on **CAMELYON16, CAMELYON17 and BRACS**, each with seeds **42, 0 and 1**, without editing the config by hand. The train/validation/test split of each dataset comes from `seed: 42` in its YAML file and is identical across runs; the run seed only changes weight initialisation and the order of training slides. Runs with saved results are skipped, so the loop can be resumed after an interruption, and the summary and figure cells can be rerun without retraining.

Outputs:

```text
results/
  {DATASET}_seed{SEED}.json                    # per-epoch history, best epoch, test metrics
  {DATASET}_seed{SEED}_test_predictions.csv    # per-slide test probabilities and predictions
  test_metrics_all_runs.csv                    # one row per run
  test_metrics_mean_std.csv                    # mean and std over seeds
  ckpt/{DATASET}_seed{SEED}.pt                 # best checkpoint (selected on validation)
  figures/
    loss_{DATASET}.png                         # train/val loss for all three seeds
    loss_{DATASET}_seed{SEED}.png              # train/val loss for one run
    attention_test_068_seed{SEED}.png          # C16 attention on the WSI, tumour slide
    attention_test_067_seed{SEED}.png          # C16 attention on the WSI, normal slide
    attention_map_{DATASET}.png                # attention on patch coordinates, all seeds
```

### Baseline test results

Mean ± standard deviation over 3 seeds (sample std). AUROC and F1 are macro-averaged over classes; accuracy is in %.

| Dataset | Classes | Test slides | AUROC | Accuracy (%) | Macro-F1 |
|---|---|---|---|---|---|
| CAMELYON16 | 2 | 129 | 0.925 ± 0.021 | 89.4 ± 3.2 | 0.881 ± 0.038 |
| CAMELYON17 | 4 | 200 | 0.848 ± 0.035 | 78.0 ± 3.9 | 0.515 ± 0.050 |
| BRACS | 7 | 86 | 0.799 ± 0.012 | 46.9 ± 1.8 | 0.374 ± 0.012 |

Per-seed values are in `results/test_metrics_all_runs.csv`. On CAMELYON16 and BRACS the best validation epoch was early (epochs 3-10 of 30); on CAMELYON17 it was later (epochs 9-22).

### Attention visualisation

- **CAMELYON16:** the provided `visualize_wsi_attention` overlays attention on `test_068` (tumour, with annotation contours) and `test_067` (normal) for each seed. The notebook saves each figure by wrapping `plt.show`, so the provided function is used as-is.
- **CAMELYON17 and BRACS:** no WSIs are included, so attention is drawn at each patch's level-0 coordinates instead. Each figure shows one confidently correct and one confidently misclassified test slide (chosen from the seed-42 predictions) across all three seeds. Colours are within-slide attention percentiles, not tumour probabilities.

### Reproducing without retraining

With the feature files in `feature/` and the provided `results/` folder:

1. Run notebook Steps 0-5 to define the model, `train_one_epoch` and `evaluate`.
2. Run the Step 8b cells. The loop finds the saved JSON files and skips training; the summary and attention cells load the saved checkpoints.

The C16 WSI overlays also need `test_067.tif`, `test_068.tif` and `test_068.xml` from the [Hugging Face dataset](https://huggingface.co/datasets/LeonYe/ECE1512-WSI-Feature) in `slide/`.
