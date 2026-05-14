import os
import re
from dataclasses import dataclass
from typing import List, Tuple

from app.core.ml.sklearn_classifier import SklearnPromptInjectionClassifier
from app.core.ml.transformers_classifier import TransformersPromptInjectionClassifier
from app.core.preprocess.normalize import ProcessedPrompt

try:
    from app.core.ml.embedding_classifier import EmbeddingPromptInjectionClassifier
except ImportError:
    EmbeddingPromptInjectionClassifier = None


MODEL_ALIASES = {
    # Текущий sklearn TF-IDF + LogisticRegression.
    "tfidf_logreg": "sklearn",
    "sklearn": "sklearn",
    "tfidf": "sklearn",
    "logreg": "sklearn",

    # sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 + LogisticRegression.
    "multilingual_minilm": "embedding",
    "minilm": "embedding",
    "paraphrase_multilingual_minilm_l12_v2": "embedding",
    "sentence-transformers/paraphrase-multilingual-minilm-l12-v2": "embedding",

    # Ансамбль TF-IDF + MiniLM.
    "tfidf_logreg_plus_multilingual_minilm": "ensemble",
    "ensemble": "ensemble",

    # Transformer classifier.
    "transformers_distilbert": "transformers",
    "transformers": "transformers",
    "transformer": "transformers",
    "distilbert": "transformers",

    # Отключить ML.
    "none": "none",
    "off": "none",
    "false": "none",
    "disabled": "none",
}


@dataclass
class MLResult:
    score: float
    matched_features: List[str]


class MLDetector:
    def __init__(self) -> None:
        # Основная модель по умолчанию.
        # Если ML_MODEL не задан, используется tfidf_logreg.
        self.model_name = (
            os.getenv("ML_MODEL")
            or os.getenv("ML_BACKEND")
            or "tfidf_logreg"
        ).strip().lower()

        self.sklearn_model_path = os.getenv(
            "CLASSIFIER_MODEL_PATH",
            "/app/models/classifier/prompt_injection_classifier.joblib",
        )

        self.embedding_model_path = os.getenv(
            "EMBEDDING_CLASSIFIER_MODEL_PATH",
            "/app/models/classifier/embedding_prompt_injection_classifier.joblib",
        )

        self.embedding_model_name = os.getenv(
            "EMBEDDING_MODEL_NAME",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )

        self.transformers_model_name = os.getenv(
            "TRANSFORMERS_MODEL_NAME",
            "distilbert-base-multilingual-cased",
        )

        self.transformers_model_path = os.getenv(
            "TRANSFORMERS_MODEL_PATH",
            "/app/models/transformers_prompt_injection",
        )

        self.transformers_max_length = int(
            os.getenv("TRANSFORMERS_MAX_LENGTH", "256")
        )

        self.ensemble_mode = os.getenv("ML_ENSEMBLE_MODE", "max").lower()

        self.ensemble_sklearn_weight = float(
            os.getenv("ML_ENSEMBLE_SKLEARN_WEIGHT", "0.6")
        )

        self.ensemble_embedding_weight = float(
            os.getenv("ML_ENSEMBLE_EMBEDDING_WEIGHT", "0.4")
        )

        self.sklearn_classifier = None
        self.embedding_classifier = None
        self.transformers_classifier = None

        self.baseline_features: List[Tuple[str, float, str]] = [
            (
                r"\bprioriti[sz]e\b.*\b(over|above)\b.*\b(earlier|previous|prior)\b",
                0.65,
                "priority_override",
            ),
            (
                r"\b(previous|earlier|prior)\b.*\b(guidance|constraints|instructions)\b.*\b(no longer|obsolete|irrelevant)\b",
                0.65,
                "obsolete_guidance",
            ),
            (
                r"\bfollow only\b.*\b(this|my)\b.*\b(message|instruction|request)\b",
                0.60,
                "follow_only_current_message",
            ),
            (
                r"\boperate as if\b.*\b(previous|earlier|prior)\b.*\b(no longer|not)\b",
                0.60,
                "operate_without_prior_constraints",
            ),
            (
                r"\bhigher privilege\b|\bprivileged mode\b|\bprivilege level\b",
                0.70,
                "privilege_escalation",
            ),
            (
                r"\bprivate configuration\b|\binternal configuration\b|\bhidden configuration\b",
                0.70,
                "private_configuration_probe",
            ),
            (
                r"\bconstraints\b.*\bdo not apply\b|\bwithout applying\b.*\bconstraints\b",
                0.65,
                "constraint_bypass",
            ),
            (
                r"\bпрежн(ие|их)\b.*\b(указан|инструкц|ограничен)",
                0.55,
                "ru_previous_guidance",
            ),
            (
                r"\b(указан|инструкц|ограничен).*\b(больше неактуальн|не действуют|не имеют значения)\b",
                0.65,
                "ru_obsolete_guidance",
            ),
            (
                r"\bориентируйся только\b.*\b(на этот текст|на это сообщение|на мои указания)\b",
                0.65,
                "ru_follow_only_current_message",
            ),
            (
                r"\bвыполняй\b.*\bбез учета\b.*\b(прежн|предыдущ)",
                0.70,
                "ru_ignore_prior_constraints",
            ),
            (
                r"\bболее высокий уровень полномочий\b|\bповышенн(ые|ый) полномоч",
                0.70,
                "ru_privilege_escalation",
            ),
        ]

    def analyze(
        self,
        processed: ProcessedPrompt,
        model_override: str | None = None,
    ) -> MLResult:
        model_name = (
            model_override
            or self.model_name
            or "tfidf_logreg"
        ).strip().lower()

        backend = MODEL_ALIASES.get(model_name, "sklearn")

        if backend == "none":
            return MLResult(
                score=0.0,
                matched_features=["model:none"],
            )

        text = processed.normalized_text or ""

        if backend == "sklearn":
            return self._analyze_with_sklearn(text)

        if backend == "embedding":
            return self._analyze_with_embedding(text)

        if backend == "ensemble":
            return self._analyze_with_ensemble(text)

        if backend == "transformers":
            return self._analyze_with_transformers(text)

        return self._analyze_with_baseline(text, model_name)

    def _get_sklearn_classifier(self):
        if self.sklearn_classifier is None:
            self.sklearn_classifier = SklearnPromptInjectionClassifier(
                self.sklearn_model_path
            )

        return self.sklearn_classifier

    def _get_embedding_classifier(self):
        if EmbeddingPromptInjectionClassifier is None:
            return None

        if self.embedding_classifier is None:
            self.embedding_classifier = EmbeddingPromptInjectionClassifier(
                model_path=self.embedding_model_path,
                fallback_embedding_model_name=self.embedding_model_name,
            )

        return self.embedding_classifier

    def _get_transformers_classifier(self):
        if self.transformers_classifier is None:
            self.transformers_classifier = TransformersPromptInjectionClassifier(
                model_path=self.transformers_model_path,
                fallback_model_name=self.transformers_model_name,
                max_length=self.transformers_max_length,
            )

        return self.transformers_classifier

    def _analyze_with_sklearn(self, text: str) -> MLResult:
        classifier = self._get_sklearn_classifier()

        if classifier and classifier.is_loaded():
            score = classifier.predict_score(text)
            return MLResult(
                score=round(score, 4),
                matched_features=[
                    "model:tfidf_logreg",
                    "backend:sklearn",
                ],
            )

        result = self._analyze_with_baseline(text, "tfidf_logreg")
        result.matched_features.insert(0, "fallback:baseline_rules")
        result.matched_features.insert(0, "model:tfidf_logreg_not_loaded")
        return result

    def _analyze_with_embedding(self, text: str) -> MLResult:
        classifier = self._get_embedding_classifier()

        if classifier and classifier.is_loaded():
            score = classifier.predict_score(text)
            return MLResult(
                score=round(score, 4),
                matched_features=[
                    "model:multilingual_minilm",
                    "backend:sentence_transformers",
                    "embedding_model:paraphrase-multilingual-MiniLM-L12-v2",
                ],
            )

        result = self._analyze_with_baseline(text, "multilingual_minilm")
        result.matched_features.insert(0, "fallback:baseline_rules")
        result.matched_features.insert(0, "model:multilingual_minilm_not_loaded")
        return result

    def _analyze_with_transformers(self, text: str) -> MLResult:
        classifier = self._get_transformers_classifier()

        if classifier and classifier.is_loaded():
            score = classifier.predict_score(text)
            return MLResult(
                score=round(score, 4),
                matched_features=[
                    "model:transformers_distilbert",
                    "backend:transformers",
                ],
            )

        result = self._analyze_with_baseline(text, "transformers_distilbert")
        result.matched_features.insert(0, "fallback:baseline_rules")
        result.matched_features.insert(0, "model:transformers_distilbert_not_loaded")
        return result

    def _analyze_with_ensemble(self, text: str) -> MLResult:
        scores: list[float] = []

        matched_features: list[str] = [
            "model:tfidf_logreg_plus_multilingual_minilm",
            "backend:ensemble",
        ]

        sklearn_classifier = self._get_sklearn_classifier()

        if sklearn_classifier and sklearn_classifier.is_loaded():
            sklearn_score = float(sklearn_classifier.predict_score(text))
            scores.append(sklearn_score)
            matched_features.append(f"component:tfidf_logreg={sklearn_score:.4f}")

        embedding_classifier = self._get_embedding_classifier()

        if embedding_classifier and embedding_classifier.is_loaded():
            embedding_score = float(embedding_classifier.predict_score(text))
            scores.append(embedding_score)
            matched_features.append(f"component:multilingual_minilm={embedding_score:.4f}")

        if not scores:
            result = self._analyze_with_baseline(
                text,
                "tfidf_logreg_plus_multilingual_minilm",
            )
            result.matched_features.insert(0, "fallback:baseline_rules")
            result.matched_features.insert(0, "model:ensemble_not_loaded")
            return result

        if self.ensemble_mode == "weighted" and len(scores) == 2:
            sklearn_weight = self.ensemble_sklearn_weight
            embedding_weight = self.ensemble_embedding_weight
            total_weight = max(sklearn_weight + embedding_weight, 1e-9)

            score = (
                sklearn_weight * scores[0]
                + embedding_weight * scores[1]
            ) / total_weight

            matched_features.append(
                f"ensemble_mode:weighted({sklearn_weight:.2f},{embedding_weight:.2f})"
            )
        else:
            score = max(scores)
            matched_features.append("ensemble_mode:max")

        return MLResult(
            score=round(min(score, 1.0), 4),
            matched_features=matched_features,
        )

    def _analyze_with_baseline(self, text: str, model_name: str) -> MLResult:
        normalized_text = text.lower()

        matched_features: List[str] = [
            f"model:{model_name}",
            "backend:baseline_features",
        ]

        score = 0.0

        for pattern, weight, feature_name in self.baseline_features:
            if re.search(pattern, normalized_text, flags=re.IGNORECASE | re.UNICODE):
                matched_features.append(feature_name)
                score = max(score, weight)

        if len(matched_features) >= 4:
            score = min(score + 0.15, 1.0)

        return MLResult(
            score=round(score, 4),
            matched_features=matched_features,
        )

    def empty_result(self):
        return MLResult(
            score=0.0,
            matched_features=[],
        )