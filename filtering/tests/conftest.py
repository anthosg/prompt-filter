import importlib.util
import sys
import types


if importlib.util.find_spec("transformers") is None:
    transformers_stub = types.ModuleType("transformers")

    class _UnavailableTransformersObject:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):  # pragma: no cover - defensive stub
            raise RuntimeError("transformers is not installed in this test environment")

    transformers_stub.AutoModelForSequenceClassification = _UnavailableTransformersObject
    transformers_stub.AutoTokenizer = _UnavailableTransformersObject
    sys.modules["transformers"] = transformers_stub
