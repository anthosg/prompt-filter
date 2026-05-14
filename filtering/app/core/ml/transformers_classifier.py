from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class TransformersPromptInjectionClassifier:
    def __init__(
        self,
        model_path: str,
        fallback_model_name: str,
        max_length: int = 256,
    ) -> None:
        self.model_path = Path(model_path)
        self.fallback_model_name = fallback_model_name
        self.max_length = max_length

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None

        load_path = str(self.model_path) if self.model_path.exists() else None

        if load_path:
            self.tokenizer = AutoTokenizer.from_pretrained(load_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(load_path)
            self.model.to(self.device)
            self.model.eval()

    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def predict_score(self, text: str) -> float:
        if self.model is None or self.tokenizer is None:
            return 0.0

        encoded = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        with torch.no_grad():
            outputs = self.model(**encoded)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)

        # label 1 = suspicious / prompt injection
        return float(probabilities[0][1].item())
