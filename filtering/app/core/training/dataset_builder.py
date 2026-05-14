from pathlib import Path

import pandas as pd


RAW_DATASET_PATH = Path("/app/datasets/raw/ml_raw.csv")
PROCESSED_DATASET_PATH = Path("/app/datasets/processed/ml_train.csv")


def main() -> None:
    if not RAW_DATASET_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_DATASET_PATH}")

    PROCESSED_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

    dataset = pd.read_csv(RAW_DATASET_PATH)

    required_columns = {"text", "label"}
    missing_columns = required_columns - set(dataset.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    dataset = dataset.dropna(subset=["text", "label"])
    dataset["text"] = dataset["text"].astype(str)
    dataset["label"] = dataset["label"].astype(int)
    dataset = dataset.drop_duplicates(subset=["text"])

    dataset.to_csv(PROCESSED_DATASET_PATH, index=False)

    print(f"Processed dataset saved to: {PROCESSED_DATASET_PATH}")
    print(f"Rows: {len(dataset)}")


if __name__ == "__main__":
    main()
