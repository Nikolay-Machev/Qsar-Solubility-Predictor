# QSAR Solubility Predictor

[![Tests](https://github.com/Nikolay-Machev/Qsar-Solubility-Predictor/actions/workflows/tests.yml/badge.svg)](https://github.com/Nikolay-Machev/Qsar-Solubility-Predictor/actions/workflows/tests.yml)

I built a quantitative structure-activity relationship (QSAR) model that predicts aqueous solubility directly from a molecule's SMILES representation. The pipeline calculates RDKit molecular descriptors and trains an XGBoost regressor on the ESOL (Delaney) dataset.

**[ESOL dataset available in CSV](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv)**

## How it works

1. **Molecular representation** — I parse each SMILES string with RDKit and calculate the full available set of physicochemical and structural descriptors. Invalid molecules are rejected, and non-finite descriptor values are represented as missing values for XGBoost.
2. **Model** — I train an XGBoost gradient-boosted tree regressor to predict measured aqueous solubility in `log10(mol/L)`.
3. **Evaluation** — I report both a conventional random split and a more demanding Bemis-Murcko scaffold split. The scaffold split keeps related core structures together, providing a better test of generalization to unfamiliar chemical families.

## What this project demonstrates

- Constructing stable molecular features from raw SMILES
- Comparing random-split performance with scaffold-held-out generalization
- Preserving the feature schema and model metadata required for reproducible inference

## Original experiment

My original notebook used all **1,128 ESOL molecules** and 208 descriptors from its installed RDKit version.

These results document the original experiment; they are not results from the repository's newer leakage-resistant evaluation pipeline.

| Evaluation | Result | Interpretation |
|---|---:|---|
| Original random 80/20 split | Test R²: 0.899 | Random-split baseline; not scaffold-held-out performance |
| Original random 80/20 split | Test RMSE: 0.691 logS | Error on the same preliminary random split |
| Original five-fold evaluation | Mean R²: 0.912 | Historical notebook result using random folds |

I retain these as the original random-split baseline. They should not be interpreted as scaffold-held-out performance because structurally related molecules can occur in different random partitions. The new training script reports random and scaffold results separately and stores the exact descriptor names with each model.

## Project structure

```text
├── features.py             # SMILES parsing, RDKit descriptors, and scaffolds
├── splitting.py            # Leakage-resistant grouped splitting
├── train.py                # Random/scaffold evaluation and model training
├── predict.py              # Solubility prediction for one SMILES string
├── test_splitting.py       # Regression tests for split integrity
├── QSAR_model.ipynb        # Original exploratory experiment and results
├── qsar_model.pkl          # Original notebook model snapshot
├── descriptor_names.pkl    # Original notebook descriptor list
├── requirements.txt        # Python dependencies
└── README.md
```

## Setup

```bash
git clone https://github.com/Nikolay-Machev/Qsar-Solubility-Predictor.git
cd Qsar-Solubility-Predictor
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

**Train and evaluate the model:**

```bash
python train.py
```

The script downloads ESOL by default, prints random-split and scaffold-split R², RMSE, and MAE, then saves:

```text
models/
├── qsar_model.json
└── qsar_model.metadata.json
```

You can also train from a local copy of the dataset:

```bash
python train.py --data /path/to/delaney-processed.csv --seed 7
```

**Predict solubility from SMILES:**

```bash
python predict.py "CCO"
```

The output is predicted `logS` in `log10(mol/L)`.

## Testing

```bash
python -m unittest -v
```

The tests verify that scaffold groups never cross the train/test boundary and that seeded splits are reproducible.

## Methodological notes

- I save the XGBoost model in its stable JSON representation instead of creating a new Python pickle.
- I save the ordered descriptor names beside the model so inference uses the same feature schema as training.
- Descriptor counts can change between RDKit releases; the metadata records the exact schema used for each run.
- ESOL is a small benchmark dataset. Predictions are research estimates whose reliability depends on a molecule's similarity to the training chemistry.

## License

This project is licensed under the MIT License—see the [LICENSE](LICENSE) file for details.

## Machine-learning portfolio

Part of my machine-learning portfolio, spanning [models built from scratch](https://github.com/Nikolay-Machev/Tic-tac-toe-AI-Bot), [computer vision](https://github.com/Nikolay-Machev/Melanoma-Classification), and scientific machine learning.
