from pathlib import Path
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import FeatureUnion, Pipeline
from app.core.ml.features import TextStatsTransformer


TRAIN_PATH = Path("/app/datasets/splits/train.csv")
VALIDATION_PATH = Path("/app/datasets/splits/validation.csv")
MODEL_PATH = Path("/app/models/classifier/prompt_injection_classifier.joblib")


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
    val_df = pd.read_csv(VALIDATION_PATH)

    print(f"Loaded train rows: {len(train_df)}", flush=True)
    print(f"Loaded validation rows: {len(val_df)}", flush=True)

    log_step("2/6 Cleaning datasets")
    train_df = clean_dataset(train_df, "Train")
    val_df = clean_dataset(val_df, "Validation")

    x_train = train_df["text"].tolist()
    y_train = train_df["label"].tolist()
    x_val = val_df["text"].tolist()
    y_val = val_df["label"].tolist()

    log_step("3/6 Building Features (TF-IDF + Meta-stats)")
    vectorizer_start = perf_counter()

    # Инициализируем TF-IDF
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=30000,
        min_df=1,
    )

    # Инициализируем наш извлекатель мета-признаков
    stats_extractor = TextStatsTransformer()

    # Объединяем их в единый пайплайн признаков
    combined_features = FeatureUnion([
        ("tfidf", vectorizer),
        ("stats", stats_extractor)
    ])

    # Обучаем и трансформируем данные через объединенный экстрактор
    x_train_vectorized = combined_features.fit_transform(x_train)

    print(f"Train matrix shape: {x_train_vectorized.shape}", flush=True)
    print(f"Feature extraction time: {perf_counter() - vectorizer_start:.2f} sec", flush=True)

    log_step("4/6 Training Logistic Regression classifier")
    train_start = perf_counter()

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        verbose=1,
    )

    classifier.fit(x_train_vectorized, y_train)

    print(f"Training stage time: {perf_counter() - train_start:.2f} sec", flush=True)

    log_step("5/6 Vectorizing validation dataset and evaluating")
    eval_start = perf_counter()

    # Используем combined_features для трансформации валидационного сета
    x_val_vectorized = combined_features.transform(x_val)
    predictions = classifier.predict(x_val_vectorized)

    print("\n=== scikit-learn validation report ===", flush=True)
    print(classification_report(y_val, predictions, zero_division=0), flush=True)

    print("=== Confusion matrix [labels: 0=benign, 1=attack] ===", flush=True)
    print(confusion_matrix(y_val, predictions, labels=[0, 1]), flush=True)

    print(f"Evaluation stage time: {perf_counter() - eval_start:.2f} sec", flush=True)

    log_step("6/6 Saving model")
    # Сохраняем полный пайплайн (Экстрактор признаков + Классификатор)
    model = Pipeline(
        steps=[
            ("features", combined_features),
            ("classifier", classifier),
        ]
    )

    joblib.dump(model, MODEL_PATH)

    print(f"Model saved to: {MODEL_PATH}", flush=True)
    print(f"Total training time: {perf_counter() - total_start:.2f} sec", flush=True)


if __name__ == "__main__":
    main()
