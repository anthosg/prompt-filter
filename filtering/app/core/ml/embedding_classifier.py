from pathlib import Path
from typing import Any

import joblib
from sentence_transformers import SentenceTransformer


class EmbeddingPromptInjectionClassifier:
    """
    ML-классификатор на основе sentence-transformers embeddings.

    Используемая модель по умолчанию:
    sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

    Схема:
    1. Текст -> embedding.
    2. Embedding -> LogisticRegression.
    3. Возвращается вероятность класса 1.
    """

    def __init__(
        self,
        model_path: str,
        fallback_embedding_model_name: str,
    ) -> None:
        self.model_path = Path(model_path)
        self.fallback_embedding_model_name = fallback_embedding_model_name

        self.encoder: SentenceTransformer | None = None
        self.classifier: Any | None = None
        self.embedding_model_name: str | None = None

        if not self.model_path.exists():
            return

        bundle = joblib.load(self.model_path)

        self.embedding_model_name = bundle.get(
            "embedding_model_name",
            self.fallback_embedding_model_name,
        )

        self.classifier = bundle.get("classifier")

        if self.classifier is not None:
            self.encoder = SentenceTransformer(self.embedding_model_name)

    def is_loaded(self) -> bool:
        return self.encoder is not None and self.classifier is not None

    def predict_score(self, text: str) -> float:
        if not self.is_loaded():
            return 0.0

        embedding = self.encoder.encode(
            [str(text)],
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        if hasattr(self.classifier, "predict_proba"):
            probabilities = self.classifier.predict_proba(embedding)[0]
            class_labels = list(self.classifier.classes_)

            if 1 in class_labels:
                positive_index = class_labels.index(1)
                return float(probabilities[positive_index])

            return 0.0

        prediction = self.classifier.predict(embedding)[0]
        return 1.0 if int(prediction) == 1 else 0.0