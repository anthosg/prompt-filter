from pathlib import Path
import shutil


SKLEARN_MODEL_PATH = Path("/app/models/classifier/prompt_injection_classifier.joblib")
SKLEARN_EXPORT_PATH = Path("/app/models/classifier/exported_prompt_injection_classifier.joblib")

TRANSFORMERS_MODEL_PATH = Path("/app/models/transformers_prompt_injection")
TRANSFORMERS_EXPORT_PATH = Path("/app/models/exported_transformers_prompt_injection")


def export_sklearn() -> None:
    if not SKLEARN_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {SKLEARN_MODEL_PATH}")

    SKLEARN_EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SKLEARN_MODEL_PATH, SKLEARN_EXPORT_PATH)

    print(f"scikit-learn model exported to: {SKLEARN_EXPORT_PATH}")


def export_transformers() -> None:
    if not TRANSFORMERS_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {TRANSFORMERS_MODEL_PATH}")

    if TRANSFORMERS_EXPORT_PATH.exists():
        shutil.rmtree(TRANSFORMERS_EXPORT_PATH)

    shutil.copytree(TRANSFORMERS_MODEL_PATH, TRANSFORMERS_EXPORT_PATH)

    print(f"Transformers model exported to: {TRANSFORMERS_EXPORT_PATH}")


def main() -> None:
    export_sklearn()

    if TRANSFORMERS_MODEL_PATH.exists():
        export_transformers()


if __name__ == "__main__":
    main()
