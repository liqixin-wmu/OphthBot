# Ophthalmic Transfer Learning Pipelines

A publication-ready, English-only repository for two related ophthalmic image-classification workflows:

- `ocular_surface`: referral-level transfer learning for ocular surface disease screening.
- `cataract`: binary cataract transfer learning with internal cross-validation and external evaluation.

This repository is designed for public release, manuscript supplements, and reproducible academic use. It removes hard-coded private paths, avoids non-ASCII identifiers, standardizes naming, and separates configuration from code.

## Highlights

- English-only code, file names, and documentation
- Command-line interfaces for both pipelines
- YAML-based configuration
- Standardized input schema with explicit column mapping support
- Fold-specific Youden-threshold selection for the ocular surface pipeline
- Stratified cross-validation and external-cohort evaluation
- Export of per-fold metrics, predictions, and ROC plots
- Safe defaults for public GitHub release

## Repository layout

```text
ophthalmic_transfer_learning_submission_repo/
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- LICENSE
|-- configs/
|   |-- ocular_surface_example.yaml
|   `-- cataract_example.yaml
|-- docs/
|   `-- data_schema.md
|-- scripts/
|   |-- run_ocular_surface.py
|   `-- run_cataract.py
`-- ophthalmic_transfer/
    |-- __init__.py
    |-- common/
    |   |-- __init__.py
    |   |-- io.py
    |   |-- metrics.py
    |   |-- modeling.py
    |   `-- utils.py
    |-- ocular_surface/
    |   |-- __init__.py
    |   |-- data.py
    |   |-- pipeline.py
    |   `-- transforms.py
    `-- cataract/
        |-- __init__.py
        |-- data.py
        |-- evaluate.py
        |-- pipeline.py
        `-- transforms.py
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Normalized input schema

This repository expects **normalized English column names**. You should rename spreadsheet columns before running the pipelines.

### Ocular surface input columns

Required columns:
- `image_path`
- `source`
- `quality_label`
- `stage1_label`
- `stage3_label`

Expected values:
- `quality_label`: `medium`, `good`, or `poor`
- `stage1_label`: `abnormal`, `normal`, or `na`
- `stage3_label`: `referral`, `non_referral`, or `na`

### Cataract input columns

Required columns:
- `image_name`
- `source`
- `label`

Expected binary labels:
- `0` = negative
- `1` = positive

Detailed schema notes are in `docs/data_schema.md`.

## Source naming convention

Use ASCII-only source identifiers, for example:

- `wenzhou`
- `hangzhou`
- `ningbo`
- `wushi`
- `aksu`
- `baicheng`
- `kuche`
- `shaya`
- `xinhe`

## Quick start

### Ocular surface pipeline

```bash
python scripts/run_ocular_surface.py --config configs/ocular_surface_example.yaml
```

### Cataract pipeline

```bash
python scripts/run_cataract.py --config configs/cataract_example.yaml
```

## Ocular surface methodological note

For each cross-validation fold, the best model is selected by validation AUROC. The saved best model is then reloaded, and the optimal threshold is computed **strictly on that fold-specific validation set** using the Youden index. That threshold is subsequently applied to the independent external test set.

This design prevents threshold leakage from training data and avoids sharing one global threshold across folds.

## Main outputs

### Ocular surface

- `fold_results.csv`
- `external_metrics_per_fold.csv`
- `external_ensemble_metrics.csv`
- `external_ensemble_predictions.csv`
- `fold_<k>_external_predictions.csv`
- `roc_validation.pdf`
- `roc_external.pdf`

### Cataract

- `cross_validation_results.csv`
- `fold_<k>_best.pth`
- `fold_<k>_history.csv`
- `roc_curves_train_val_external.png`
- `fold_metrics.csv`
- `fold_metrics_summary.csv`
- `all_fold_results.pkl`
- `fold_<k>_<stage>.csv`

## Suggested manuscript wording

### Ocular surface threshold selection

> In each cross-validation fold, the model with the highest validation-set AUROC was selected and saved. The selected model was then reloaded, and the optimal decision threshold was determined exclusively on the corresponding validation set using the Youden index. This fold-specific threshold was subsequently applied to the independent external test set.

### Cataract transfer learning

> A DenseNet121-based transfer-learning framework was used for binary cataract classification. In each fold of stratified cross-validation, model training was performed on the internal cohort with class balancing and weighted loss, while evaluation was conducted on the corresponding validation split and on an independent external cohort defined by source sites.