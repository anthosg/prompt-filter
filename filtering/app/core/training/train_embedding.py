import os
import torch
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


TRAIN_PATH = Path(os.getenv("TRAIN_PATH", "/app/datasets/splits/train.csv"))
VALIDATION_PATH = Path(os.getenv("VALIDATION_PATH", "/app/datasets/splits/validation.csv"))

MODEL_PATH = Path(
    os.getenv(
        "EMBEDDING_CLASSIFIER_MODEL_PATH",
        "/app/models/classifier/embedding_prompt_injection_classifier.joblib",
    )
)

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))


def log_step(message: str) -> None:
    print(f"\n=== {message} ===", flush=True)


def clean_dataset(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    required_columns = {"text", "label"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{dataset_name} dataset must contain columns: {sorted(required_columns)}. "
            f"Missing: {sorted(missing_columns)}"
        )

    before = len(df)

    df = df.copy()
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1])]

    df = df.drop_duplicates(subset=["text"])

    after = len(df)

    print(f"{dataset_name}: rows before cleaning = {before}", flush=True)
    print(f"{dataset_name}: rows after cleaning  = {after}", flush=True)
    print(f"{dataset_name}: removed rows         = {before - after}", flush=True)

    if df.empty:
        raise ValueError(f"{dataset_name} dataset is empty after cleaning")

    if df["label"].nunique() < 2:
        raise ValueError(
            f"{dataset_name} dataset must contain both classes 0 and 1 after cleaning"
        )

    print(f"\n{dataset_name} label distribution:", flush=True)
    print(df["label"].value_counts().sort_index(), flush=True)

    return df


def main() -> None:
    total_start = perf_counter()

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Train dataset not found: {TRAIN_PATH}")

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(f"Validation dataset not found: {VALIDATION_PATH}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    log_step("1/6 Loading datasets")
    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)

    print(f"Loaded train rows: {len(train_df)}", flush=True)
    print(f"Loaded validation rows: {len(validation_df)}", flush=True)

    log_step("2/6 Cleaning datasets")
    train_df = clean_dataset(train_df, "Train")
    validation_df = clean_dataset(validation_df, "Validation")

    train_texts = train_df["text"].astype(str).tolist()
    train_labels = train_df["label"].astype(int).tolist()

    validation_texts = validation_df["text"].astype(str).tolist()
    validation_labels = validation_df["label"].astype(int).tolist()

    log_step(f"3/6 Loading embedding model: {EMBEDDING_MODEL_NAME}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)
    encoder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

    log_step("4/6 Encoding train and validation texts")
    encode_start = perf_counter()

    train_embeddings = encoder.encode(
        train_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    validation_embeddings = encoder.encode(
        validation_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    print(f"Train embedding shape: {train_embeddings.shape}", flush=True)
    print(f"Validation embedding shape: {validation_embeddings.shape}", flush=True)
    print(f"Encoding time: {perf_counter() - encode_start:.2f} sec", flush=True)

    log_step("5/6 Training LogisticRegression on embeddings")
    train_start = perf_counter()

    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )

    classifier.fit(train_embeddings, train_labels)

    print(f"Training time: {perf_counter() - train_start:.2f} sec", flush=True)

    log_step("6/6 Evaluating and saving model")

    predictions = classifier.predict(validation_embeddings)

    print("\n=== Embedding classifier validation report ===", flush=True)
    print(classification_report(validation_labels, predictions, zero_division=0), flush=True)

    print("=== Confusion matrix [labels: 0=benign, 1=attack] ===", flush=True)
    print(confusion_matrix(validation_labels, predictions, labels=[0, 1]), flush=True)

    bundle = {
        "embedding_model_name": EMBEDDING_MODEL_NAME,
        "classifier": classifier,
    }

    joblib.dump(bundle, MODEL_PATH)

    print(f"Model saved to: {MODEL_PATH}", flush=True)
    print(f"Total training time: {perf_counter() - total_start:.2f} sec", flush=True)


if __name__ == "__main__":
    main()