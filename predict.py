"""Predict aqueous solubility from a SMILES string."""

import argparse
import json
from pathlib import Path

from xgboost import XGBRegressor

from features import calculate_descriptors


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("smiles")
    parser.add_argument("--model", type=Path, default=Path("models/qsar_model.json"))
    return parser.parse_args()


def main():
    args = parse_args()
    metadata_path = args.model.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    descriptors, _ = calculate_descriptors(args.smiles, metadata["descriptor_names"])
    model = XGBRegressor()
    model.load_model(str(args.model))
    prediction = float(model.predict(descriptors.reshape(1, -1))[0])
    print(f"Predicted logS: {prediction:.3f} log10(mol/L)")
    print("Research estimate only; applicability depends on similarity to ESOL chemistry.")


if __name__ == "__main__":
    main()
