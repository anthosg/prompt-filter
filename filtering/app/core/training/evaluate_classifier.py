import os
from pathlib import Path

import joblib
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer


TEST_DATASET_PATH = Path(
    os.getenv(
        "TEST_DATASET_PATH",
        "/app/datasets/splits/test.csv",
    )
)

SKLEARN_MODEL_PATH = Path(
    os.getenv(
        "CLASSIFIER_MODEL_PATH",
        "/app/models/classifier/prompt_injection_classifier.joblib",
    )
)

TRANSFORMERS_MODEL_PATH = Path(
    os.getenv(
        "TRANSFORMERS_MODEL_PATH",
        "/app/models/transformers_prompt_injection",
    )
)

ML_BACKEND = os.getenv("ML_BACKEND", "sklearn").lower()
MAX_LENGTH = int(os.getenv("TRANSFORMERS_MAX_LENGTH", "256"))


def print_metrics(y_true, y_pred) -> None:
    print(f"Dataset: {TEST_DATASET_PATH}")
    print(f"Rows: {len(y_true)}")

    print("\n=== Evaluation report ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print("Confusion matrix [labels: 0=benign, 1=attack]:")
    print(matrix)

    tn, fp, fn, tp = matrix.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    print("\n=== Binary metrics ===")
    print(f"TP: {tp}")
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"FPR:       {fpr:.4f}")
    print(f"FNR:       {fnr:.4f}")


def evaluate_sklearn(dataset: pd.DataFrame) -> None:
    if not SKLEARN_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {SKLEARN_MODEL_PATH}")

    model = joblib.load(SKLEARN_MODEL_PATH)
    predictions = model.predict(dataset["text"])

    print("=== scikit-learn test evaluation ===")
    print_metrics(dataset["label"], predictions)


def evaluate_transformers(dataset: pd.DataFrame) -> None:
    if not TRANSFORMERS_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {TRANSFORMERS_MODEL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(str(TRANSFORMERS_MODEL_PATH))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(TRANSFORMERS_MODEL_PATH)
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    predictions = []

    for text in dataset["text"].tolist():
        encoded = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        with torch.no_grad():
            outputs = model(**encoded)
            prediction = torch.argmax(outputs.logits, dim=-1).item()

        predictions.append(prediction)

    print("=== Transformers test evaluation ===")
    print_metrics(dataset["label"], predictions)


def main() -> None:
    if not TEST_DATASET_PATH.exists():
        raise FileNotFoundError(f"Test dataset not found: {TEST_DATASET_PATH}")

    dataset = pd.read_csv(TEST_DATASET_PATH)

    if "text" not in dataset.columns or "label" not in dataset.columns:
        raise ValueError("Dataset must contain columns: text,label")

    dataset = dataset.dropna(subset=["text", "label"])
    dataset["text"] = dataset["text"].astype(str)
    dataset["label"] = dataset["label"].astype(int)

    print("Label distribution:")
    print(dataset["label"].value_counts().sort_index())

    if "source" in dataset.columns:
        print("\nSource distribution:")
        print(dataset["source"].value_counts())

    if ML_BACKEND == "sklearn":
        evaluate_sklearn(dataset)
    elif ML_BACKEND == "transformers":
        evaluate_transformers(dataset)
    else:
        raise ValueError(f"Unsupported ML_BACKEND: {ML_BACKEND}")


if __name__ == "__main__":
    main()