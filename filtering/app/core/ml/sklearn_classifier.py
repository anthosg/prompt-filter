from pathlib import Path
import joblib
from app.core.ml.features import TextStatsTransformer


class SklearnPromptInjectionClassifier:
    def __init__(self, model_path: str) -> None:
        self.model_path = Path(model_path)
        self.model = None

        if self.model_path.exists():
            self.model = joblib.load(self.model_path)

    def is_loaded(self) -> bool:
        return self.model is not None

    def predict_score(self, text: str) -> float:
        if self.model is None:
            return 0.0

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba([text])[0]
            class_labels = list(self.model.classes_)

            if 1 in class_labels:
                suspicious_index = class_labels.index(1)
                return float(probabilities[suspicious_index])

            return 0.0

        prediction = self.model.predict([text])[0]
        return 1.0 if int(prediction) == 1 else 0.0
