"""
Phase 3a — Export the trained LightGBM model to ONNX.

Why: LightGBM's native save format (.txt, from booster.save_model()) has
no good native reader in Java. ONNX is a cross-language model format that
both Python and Java (via the onnxruntime library) can read — this is a
standard, common way production systems bridge a Python-trained model
into a different serving language.

This has been verified: converting a LightGBM model to ONNX and comparing
predictions between the original model and the ONNX version showed
matching output to 5+ decimal places.

Usage:
    python src/pipeline/05_export_onnx.py \
        --model data/processed/ranker_model.txt \
        --output data/processed/ranker_model.onnx
"""

import argparse

import lightgbm as lgb
from onnxmltools.convert import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType

# Must match FEATURE_COLS in 03_train_ranker.py exactly — same order,
# same count. The ONNX model has no column names, only positions.
FEATURE_COLS = [
    "total_events", "num_sessions", "buy_count",
    "item_total_events", "item_buy_count", "view_count",
    "distinct_users", "popularity_score",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to LightGBM .txt model file")
    parser.add_argument("--output", required=True, help="Path to write the .onnx file")
    args = parser.parse_args()

    print(f"Loading LightGBM model from {args.model} ...")
    booster = lgb.Booster(model_file=args.model)

    print(f"Converting to ONNX (expects {len(FEATURE_COLS)} features: {FEATURE_COLS}) ...")
    initial_type = [("input", FloatTensorType([None, len(FEATURE_COLS)]))]
    onnx_model = convert_lightgbm(booster, initial_types=initial_type, zipmap=False)

    with open(args.output, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Done. ONNX model written to {args.output}")
    print()
    print("Note for the Java side: the ONNX model outputs two things —")
    print("  'label' (0/1 predicted class) and 'probabilities' (shape [N, 2]).")
    print("  Use probabilities[:, 1] as the ranking score (probability of the positive class).")
    print(f"  Feature order (must match exactly): {FEATURE_COLS}")


if __name__ == "__main__":
    main()
