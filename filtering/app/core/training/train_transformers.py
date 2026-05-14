import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


TRAIN_PATH = Path("/app/datasets/splits/train.csv")
VALIDATION_PATH = Path("/app/datasets/splits/validation.csv")

MODEL_OUTPUT_PATH = Path(
    os.getenv("TRANSFORMERS_MODEL_PATH", "/app/models/transformers_prompt_injection")
)

BASE_MODEL_NAME = os.getenv(
    "TRANSFORMERS_MODEL_NAME",
    "distilbert-base-multilingual-cased",
)

MAX_LENGTH = int(os.getenv("TRANSFORMERS_MAX_LENGTH", "256"))


class PromptInjectionDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int = 256):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            key: value.squeeze(0)
            for key, value in encoded.items()
        }

        item["labels"] = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )

    accuracy = accuracy_score(labels, predictions)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> None:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Train dataset not found: {TRAIN_PATH}")

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(f"Validation dataset not found: {VALIDATION_PATH}")

    MODEL_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VALIDATION_PATH)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    train_dataset = PromptInjectionDataset(
        train_df["text"],
        train_df["label"],
        tokenizer,
        MAX_LENGTH,
    )

    validation_dataset = PromptInjectionDataset(
        val_df["text"],
        val_df["label"],
        tokenizer,
        MAX_LENGTH,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=2,
    )

    training_args = TrainingArguments(
        output_dir="/app/models/transformers_training_output",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="/app/models/transformers_training_logs",
        logging_steps=20,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    metrics = trainer.evaluate()
    print("=== Transformers validation metrics ===")
    print(metrics)

    trainer.save_model(str(MODEL_OUTPUT_PATH))
    tokenizer.save_pretrained(str(MODEL_OUTPUT_PATH))

    print(f"Transformers model saved to: {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()