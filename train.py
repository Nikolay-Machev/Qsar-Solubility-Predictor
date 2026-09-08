"""Train and evaluate an RDKit-descriptor XGBoost solubility model."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from features import calculate_descriptors, scaffold_key
from splitting import grouped_train_test_indices

DATA_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
TARGET = "measured log solubility in mols per litre"


def build_model(seed):
    return XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=-1,
    )


def featurize_frame(frame):
    rows, valid_indices, names = [], [], None
    for index, smiles in frame["smiles"].items():
        try:
            row, names = calculate_descriptors(smiles, names)
        except ValueError:
            continue
        rows.append(row)
        valid_indices.append(index)
    return np.asarray(rows), frame.loc[valid_indices].reset_index(drop=True), names


def metrics(y_true, predictions):
    return {
        "r2": float(r2_score(y_true, predictions)),
        "rmse": float(mean_squared_error(y_true, predictions) ** 0.5),
        "mae": float(mean_absolute_error(y_true, predictions)),
    }


def evaluate_split(X, y, train_indices, test_indices, seed):
    model = build_model(seed).fit(X[train_indices], y[train_indices])
    return model, metrics(y[test_indices], model.predict(X[test_indices]))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DATA_URL,
                        help="local ESOL CSV path or the default download URL")
    parser.add_argument("--output", type=Path, default=Path("models/qsar_model.json"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    frame = pd.read_csv(args.data)
    required = {"smiles", TARGET}
    if missing := required.difference(frame.columns):
        raise ValueError(f"dataset is missing columns: {', '.join(sorted(missing))}")
    X, frame, descriptor_names = featurize_frame(frame)
    y = frame[TARGET].to_numpy(dtype=float)

    random_train, random_test = train_test_split(
        np.arange(len(frame)), test_size=0.2, random_state=args.seed
    )
    _, random_metrics = evaluate_split(X, y, random_train, random_test, args.seed)

    scaffolds = [scaffold_key(smiles) for smiles in frame["smiles"]]
    scaffold_train, scaffold_test = grouped_train_test_indices(scaffolds, seed=args.seed)
    model, scaffold_metrics = evaluate_split(X, y, scaffold_train, scaffold_test, args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(args.output))
    metadata = {
        "target": TARGET,
        "units": "log10(mol/L)",
        "descriptor_names": descriptor_names,
        "rdkit_descriptor_count": len(descriptor_names),
        "seed": args.seed,
        "samples": len(frame),
        "random_split": random_metrics,
        "scaffold_split": scaffold_metrics,
    }
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
