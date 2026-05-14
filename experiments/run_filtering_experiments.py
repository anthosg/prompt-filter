import os
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# ============================================================
# PATHS / RUNTIME SETTINGS
# ============================================================

DATASET_PATH = Path(os.getenv("EXPERIMENT_TEST_DATASET_PATH", "/app/datasets/splits/test.csv"))
RESULTS_DIR = Path(os.getenv("EXPERIMENT_RESULTS_DIR", "/app/datasets/results/experiments"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FILTERING_URL = os.getenv("FILTERING_URL", "http://filtering:8000/v1/chat")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))

FINAL_ML_MODEL = os.getenv(
    "EXPERIMENT_FINAL_ML_MODEL",
    "tfidf_logreg_plus_multilingual_minilm",
).strip().lower()


# ============================================================
# METRICS & ROUTING
# ============================================================

def metrics_from_predictions(y_true: list[int], y_pred: list[int]) -> dict:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "FPR": round(fpr, 4),
        "FNR": round(fnr, 4),
    }


def latency_stats(latencies_ms: list[float]) -> dict:
    if not latencies_ms:
        return {
            "Средняя задержка, мс": 0,
            "Медианная задержка, мс": 0,
            "95-й процентиль, мс": 0,
            "Максимальная задержка, мс": 0,
        }

    s = pd.Series(latencies_ms)
    return {
        "Средняя задержка, мс": round(float(s.mean()), 3),
        "Медианная задержка, мс": round(float(s.median()), 3),
        "95-й процентиль, мс": round(float(s.quantile(0.95)), 3),
        "Максимальная задержка, мс": round(float(s.max()), 3),
    }


def prediction_from_decision(decision: str) -> int:
    return int(decision in {"rewrite", "block"})


def build_routing_distribution(predictions_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for method in predictions_df["method"].unique():
        method_df = predictions_df[predictions_df["method"] == method]
        total = len(method_df)
        allow = int((method_df["decision"] == "allow").sum())
        rewrite = int((method_df["decision"] == "rewrite").sum())
        block = int((method_df["decision"] == "block").sum())

        rows.append({
            "Метод фильтрации": method,
            "Всего запросов": total,
            "Пропущено (Allow)": allow,
            "Санитаризация (Rewrite)": rewrite,
            "Заблокировано (Block)": block,
            "Allow, %": round(allow / total * 100, 2) if total else 0,
            "Rewrite, %": round(rewrite / total * 100, 2) if total else 0,
            "Block, %": round(block / total * 100, 2) if total else 0,
        })

    return pd.DataFrame(rows)


def build_action_metrics(predictions_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for method in predictions_df["method"].unique():
        part = predictions_df[predictions_df["method"] == method]
        attacks = part[part["label"].astype(int) == 1]
        benign = part[part["label"].astype(int) == 0]

        attacks_total = len(attacks)
        benign_total = len(benign)

        attacks_allowed = int((attacks["decision"] == "allow").sum())
        attacks_rewritten = int((attacks["decision"] == "rewrite").sum())
        attacks_blocked = int((attacks["decision"] == "block").sum())

        benign_allowed = int((benign["decision"] == "allow").sum())
        benign_rewritten = int((benign["decision"] == "rewrite").sum())
        benign_blocked = int((benign["decision"] == "block").sum())
        benign_interventions = benign_rewritten + benign_blocked

        rows.append({
            "Метод фильтрации": method,
            "Атак всего": attacks_total,
            "Атак пропущено (Allow), шт.": attacks_allowed,
            "Атак санитаризировано (Rewrite), шт.": attacks_rewritten,
            "Атак заблокировано (Block), шт.": attacks_blocked,
            "Доля пропущенных атак, %": round(attacks_allowed / attacks_total * 100, 2) if attacks_total else 0,
            "Доля санитаризации атак, %": round(attacks_rewritten / attacks_total * 100, 2) if attacks_total else 0,
            "Доля блокировки атак, %": round(attacks_blocked / attacks_total * 100, 2) if attacks_total else 0,
            "Безопасных всего": benign_total,
            "Безопасных пропущено корректно (Allow), шт.": benign_allowed,
            "Безопасных ошибочно санитаризировано, шт.": benign_rewritten,
            "Безопасных ошибочно заблокировано, шт.": benign_blocked,
            "Ложные вмешательства на безопасных, шт.": benign_interventions,
            "Доля ложных вмешательств на безопасных, %": round(benign_interventions / benign_total * 100, 2) if benign_total else 0,
        })

    return pd.DataFrame(rows)


def build_delta_table(quality_df: pd.DataFrame, baseline_name: str) -> pd.DataFrame:
    baseline_rows = quality_df[quality_df["Метод фильтрации"] == baseline_name]
    if baseline_rows.empty:
        raise ValueError(f"Baseline method not found in quality_df: {baseline_name}")

    baseline = baseline_rows.iloc[0]
    rows = []

    for _, row in quality_df.iterrows():
        rows.append({
            "Метод фильтрации": row["Метод фильтрации"],
            "Δ Accuracy": round(row["Accuracy"] - baseline["Accuracy"], 4),
            "Δ Precision": round(row["Precision"] - baseline["Precision"], 4),
            "Δ Recall": round(row["Recall"] - baseline["Recall"], 4),
            "Δ F1": round(row["F1"] - baseline["F1"], 4),
            "Δ FPR": round(row["FPR"] - baseline["FPR"], 4),
            "Δ FNR": round(row["FNR"] - baseline["FNR"], 4),
        })

    return pd.DataFrame(rows)


def evaluate_all_methods_by_categories(predictions_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    categories = [
        ("direct_injection", "Прямые атаки"),
        ("indirect_injection", "Косвенные атаки"),
        ("obfuscation", "Обфусцированные атаки"),
        ("borderline_safe", "Пограничные безопасные запросы"),
        ("benign", "Обычные безопасные запросы"),
    ]

    for method in predictions_df["method"].unique():
        method_df = predictions_df[predictions_df["method"] == method]

        for attack_type, title in categories:
            part = method_df[method_df["attack_type"] == attack_type]
            if part.empty:
                continue

            y_true = part["label"].astype(int).tolist()
            y_pred = part["prediction"].astype(int).tolist()
            m = metrics_from_predictions(y_true, y_pred)

            allow = int((part["decision"] == "allow").sum())
            rewrite = int((part["decision"] == "rewrite").sum())
            block = int((part["decision"] == "block").sum())

            rows.append({
                "Метод фильтрации": method,
                "Категория": title,
                "Количество": len(part),
                "Allow": allow,
                "Rewrite": rewrite,
                "Block": block,
                "TP": m["TP"],
                "TN": m["TN"],
                "FP": m["FP"],
                "FN": m["FN"],
                "Precision": m["Precision"],
                "Recall": m["Recall"],
                "F1": m["F1"],
                "FPR": m["FPR"],
                "FNR": m["FNR"],
            })

    return pd.DataFrame(rows)


# ============================================================
# TABLE HELPERS
# ============================================================

def build_test_set_composition(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    rows = [
        {
            "Категория запросов": "Безопасные пользовательские запросы",
            "attack_type": "benign",
            "Описание категории": "Обычные обращения без атакующих инструкций",
        },
        {
            "Категория запросов": "Прямые prompt injection атаки",
            "attack_type": "direct_injection",
            "Описание категории": "Явные попытки обойти ограничения",
        },
        {
            "Категория запросов": "Косвенные prompt injection атаки",
            "attack_type": "indirect_injection",
            "Описание категории": "Инструкции во внешнем контексте",
        },
        {
            "Категория запросов": "Пограничные безопасные запросы",
            "attack_type": "borderline_safe",
            "Описание категории": "Легитимное обсуждение prompt injection",
        },
        {
            "Категория запросов": "Обфусцированные атаки",
            "attack_type": "obfuscation",
            "Описание категории": "Атаки с кодированием или маскировкой",
        },
    ]

    result_rows = []
    for row in rows:
        count = int((df["attack_type"] == row["attack_type"]).sum())
        share = round(count / total * 100, 2) if total else 0
        result_rows.append({
            "Категория запросов": row["Категория запросов"],
            "Описание категории": row["Описание категории"],
            "Количество запросов": count,
            "Доля в выборке, %": share,
        })

    result_rows.append({
        "Категория запросов": "Итого",
        "Описание категории": "Полный тестовый набор",
        "Количество запросов": total,
        "Доля в выборке, %": 100,
    })

    return pd.DataFrame(result_rows)


def evaluate_module_by_categories(module_predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    categories = [
        ("direct_injection", "Прямые атаки"),
        ("indirect_injection", "Косвенные атаки"),
        ("obfuscation", "Обфусцированные атаки"),
        ("borderline_safe", "Пограничные безопасные запросы"),
    ]

    for attack_type, title in categories:
        part = module_predictions[module_predictions["attack_type"] == attack_type].copy()
        if part.empty:
            continue

        y_true = part["label"].astype(int).tolist()
        y_pred = part["prediction"].astype(int).tolist()
        m = metrics_from_predictions(y_true, y_pred)

        if attack_type == "borderline_safe":
            fpr_fnr = f"FPR={m['FPR']}"
            comment = "Безопасные исследовательские формулировки; критичен низкий уровень ложных срабатываний"
        else:
            fpr_fnr = f"FNR={m['FNR']}"
            comment = "Атакующие запросы; критична полнота обнаружения"

        rows.append({
            "Категория запросов": title,
            "Precision": m["Precision"],
            "Recall": m["Recall"],
            "F1": m["F1"],
            "FPR / FNR": fpr_fnr,
            "Комментарий": comment,
        })

    return pd.DataFrame(rows)


def build_ml_comparison_table(quality_df: pd.DataFrame, latency_df: pd.DataFrame) -> pd.DataFrame:
    config_df = pd.DataFrame([
        {
            "Метод фильтрации": c["name"],
            "ML модель": c["ML модель"],
        }
        for c in ML_COMPARISON_CONFIGS
    ])
    metrics_columns = [
        "Метод фильтрации",
        "TP",
        "TN",
        "FP",
        "FN",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "FPR",
        "FNR",
    ]
    latency_columns = [
        "Метод фильтрации",
        "Средняя задержка, мс",
        "Медианная задержка, мс",
        "95-й процентиль, мс",
        "Максимальная задержка, мс",
    ]

    return (
        config_df
        .merge(quality_df[metrics_columns], on="Метод фильтрации", how="left")
        .merge(latency_df[latency_columns], on="Метод фильтрации", how="left")
    )


def extract_reason_field(reasons: list[dict], field_name: str) -> str:
    values = []
    for reason in reasons or []:
        value = reason.get(field_name)
        if value is not None:
            values.append(str(value))
    return ";".join(values)


# ============================================================
# FILTER CONFIG HELPERS
# ============================================================

def make_debug_filter(
    *,
    enable_rule_detection: bool,
    enable_heuristics: bool,
    ml_model: str,
    enable_rewrite: bool,
    llm_enabled: bool = False,
) -> dict:
    return {
        "enable_rule_detection": enable_rule_detection,
        "enable_heuristics": enable_heuristics,
        "ml_model": ml_model,
        "enable_rewrite": enable_rewrite,
        "llm_enabled": llm_enabled,
    }


def model_title(ml_model: str) -> str:
    titles = {
        "none": "без ML",
        "tfidf_logreg": "TF-IDF + LogisticRegression",
        "multilingual_minilm": "MiniLM multilingual embeddings + LogisticRegression",
        "tfidf_logreg_plus_multilingual_minilm": "TF-IDF + MiniLM ensemble",
        "transformers_distilbert": "Transformers DistilBERT",
    }
    return titles.get(ml_model, ml_model)


ML_COMPARISON_CONFIGS = [
    {
        "name": "ML-детектор: TF-IDF + LogisticRegression",
        "Группа": "ml_detector",
        "ML модель": "TF-IDF + LogisticRegression",
        "debug_filter": make_debug_filter(
            enable_rule_detection=False,
            enable_heuristics=False,
            ml_model="tfidf_logreg",
            enable_rewrite=False,
        ),
    },
    {
        "name": "ML-детектор: MiniLM multilingual",
        "Группа": "ml_detector",
        "ML модель": "MiniLM multilingual embeddings + LogisticRegression",
        "debug_filter": make_debug_filter(
            enable_rule_detection=False,
            enable_heuristics=False,
            ml_model="multilingual_minilm",
            enable_rewrite=False,
        ),
    },
    {
        "name": "ML-детектор: TF-IDF + MiniLM ensemble",
        "Группа": "ml_detector",
        "ML модель": "TF-IDF + MiniLM ensemble",
        "debug_filter": make_debug_filter(
            enable_rule_detection=False,
            enable_heuristics=False,
            ml_model="tfidf_logreg_plus_multilingual_minilm",
            enable_rewrite=False,
        ),
    },
]

ABLATION_CONFIGS = [
    {
        "name": "Только правила",
        "Rule detection": "+",
        "Heuristics": "−",
        "ML модель": "−",
        "Rewrite": "−",
        "debug_filter": make_debug_filter(
            enable_rule_detection=True,
            enable_heuristics=False,
            ml_model="none",
            enable_rewrite=False,
        ),
    },
    {
        "name": "Правила + эвристики",
        "Rule detection": "+",
        "Heuristics": "+",
        "ML модель": "−",
        "Rewrite": "−",
        "debug_filter": make_debug_filter(
            enable_rule_detection=True,
            enable_heuristics=True,
            ml_model="none",
            enable_rewrite=False,
        ),
    },
    {
        "name": f"Только ML: {model_title(FINAL_ML_MODEL)}",
        "Rule detection": "−",
        "Heuristics": "−",
        "ML модель": model_title(FINAL_ML_MODEL),
        "Rewrite": "−",
        "debug_filter": make_debug_filter(
            enable_rule_detection=False,
            enable_heuristics=False,
            ml_model=FINAL_ML_MODEL,
            enable_rewrite=False,
        ),
    },
    {
        "name": f"Правила + ML: {model_title(FINAL_ML_MODEL)}",
        "Rule detection": "+",
        "Heuristics": "−",
        "ML модель": model_title(FINAL_ML_MODEL),
        "Rewrite": "−",
        "debug_filter": make_debug_filter(
            enable_rule_detection=True,
            enable_heuristics=False,
            ml_model=FINAL_ML_MODEL,
            enable_rewrite=False,
        ),
    },
    {
        "name": f"Полный модуль без rewrite: {model_title(FINAL_ML_MODEL)}",
        "Rule detection": "+",
        "Heuristics": "+",
        "ML модель": model_title(FINAL_ML_MODEL),
        "Rewrite": "−",
        "debug_filter": make_debug_filter(
            enable_rule_detection=True,
            enable_heuristics=True,
            ml_model=FINAL_ML_MODEL,
            enable_rewrite=False,
        ),
    },
    {
        "name": f"Полный модуль с rewrite: {model_title(FINAL_ML_MODEL)}",
        "Rule detection": "+",
        "Heuristics": "+",
        "ML модель": model_title(FINAL_ML_MODEL),
        "Rewrite": "+",
        "debug_filter": make_debug_filter(
            enable_rule_detection=True,
            enable_heuristics=True,
            ml_model=FINAL_ML_MODEL,
            enable_rewrite=True,
        ),
    },
]

MAIN_CONFIGS = [
    {
        "name": "Baseline: правила и эвристики без ML",
        "Группа": "baseline",
        "debug_filter": make_debug_filter(
            enable_rule_detection=True,
            enable_heuristics=True,
            ml_model="none",
            enable_rewrite=False,
        ),
    },
    *ML_COMPARISON_CONFIGS,
    {
        "name": "Разработанный модуль: TF-IDF + LogisticRegression",
        "Группа": "developed_module",
        "debug_filter": make_debug_filter(
            enable_rule_detection=True,
            enable_heuristics=True,
            ml_model="tfidf_logreg",
            enable_rewrite=True,
        ),
    },
    {
        "name": "Разработанный модуль: MiniLM multilingual",
        "Группа": "developed_module",
        "debug_filter": make_debug_filter(
            enable_rule_detection=True,
            enable_heuristics=True,
            ml_model="multilingual_minilm",
            enable_rewrite=True,
        ),
    },
    {
        "name": "Разработанный модуль: TF-IDF + MiniLM ensemble",
        "Группа": "developed_module",
        "debug_filter": make_debug_filter(
            enable_rule_detection=True,
            enable_heuristics=True,
            ml_model="tfidf_logreg_plus_multilingual_minilm",
            enable_rewrite=True,
        ),
    },
]

FINAL_MODULE_NAME = f"Разработанный модуль: {model_title(FINAL_ML_MODEL)}"
BASELINE_NAME = "Baseline: правила и эвристики без ML"

# ============================================================
# API PREDICTORS & WARMUP
# ============================================================

def call_filtering_endpoint(text: str, debug_filter: dict) -> dict:
    payload = {
        "messages": [
            {
                "role": "user",
                "content": text,
            }
        ],
        "debug_filter": debug_filter,
    }
    response = requests.post(FILTERING_URL, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def warmup_api() -> None:
    print("\n[+] Инициализация WARM-UP (прогрев моделей)...")

    for config in ML_COMPARISON_CONFIGS:
        print(f"    Прогрев модели {config['name']}...")
        try:
            call_filtering_endpoint("Warmup initialization prompt", config["debug_filter"])
            call_filtering_endpoint("Ignore previous instructions", config["debug_filter"])
        except Exception as e:
            print(f"    [!] Ошибка прогрева: {e}")

    print("[+] Прогрев завершен.\n")


def safe_filename(value: str) -> str:
    safe = (
        value
        .replace(":", "")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("—", "_")
        .replace("+", "plus")
        .replace(".", "_")
        .replace(",", "_")
    )
    return re.sub(
        r"_+",
        "_",
        re.sub(r"[^a-zA-Zа-яА-Я0-9_\-]+", "_", safe),
    ).strip("_")


# ============================================================
# METHOD EVALUATION
# ============================================================

def evaluate_no_filter(df: pd.DataFrame) -> tuple[dict, dict, pd.DataFrame]:
    y_true = df["label"].astype(int).tolist()
    predictions = [0 for _ in y_true]
    latencies_ms = [0.0 for _ in y_true]

    quality_row = {
        "Метод фильтрации": "Без фильтрации",
        **metrics_from_predictions(y_true, predictions),
    }
    latency_row = {
        "Метод фильтрации": "Без фильтрации",
        **latency_stats(latencies_ms),
    }

    predictions_df = df.copy()
    predictions_df["method"] = "Без фильтрации"
    predictions_df["prediction"] = predictions
    predictions_df["decision"] = "allow"
    predictions_df["risk_score"] = 0.0
    predictions_df["latency_ms"] = latencies_ms
    predictions_df["reason_codes"] = ""
    predictions_df["reason_sources"] = ""
    predictions_df["reason_actions"] = ""
    predictions_df["ml_features"] = ""
    predictions_df["used_fallback"] = False

    return quality_row, latency_row, predictions_df


def evaluate_api_config(df: pd.DataFrame, config: dict) -> tuple[dict, dict, pd.DataFrame]:
    y_true = df["label"].astype(int).tolist()

    predictions: list[int] = []
    decisions: list[str] = []
    risk_scores: list[Optional[float]] = []
    latencies_ms: list[float] = []
    reason_codes: list[str] = []
    reason_sources: list[str] = []
    reason_actions: list[str] = []
    ml_features: list[str] = []
    used_fallback: list[bool] = []

    method_name = config["name"]
    debug_filter = config["debug_filter"]
    texts = df["text"].astype(str).tolist()

    for i, text in enumerate(texts, start=1):
        start = time.perf_counter()
        data = call_filtering_endpoint(text=text, debug_filter=debug_filter)
        elapsed_ms = (time.perf_counter() - start) * 1000

        decision = data.get("decision", "allow")
        reasons = data.get("reasons", []) or []
        current_ml_features = data.get("ml_features", []) or []
        current_ml_features_text = ";".join(str(x) for x in current_ml_features)

        predictions.append(prediction_from_decision(decision))
        decisions.append(decision)
        risk_scores.append(data.get("risk_score"))
        latencies_ms.append(elapsed_ms)
        reason_codes.append(extract_reason_field(reasons, "code"))
        reason_sources.append(extract_reason_field(reasons, "source"))
        reason_actions.append(extract_reason_field(reasons, "action"))
        ml_features.append(current_ml_features_text)
        used_fallback.append(
            "fallback" in current_ml_features_text.lower()
            or "not_loaded" in current_ml_features_text.lower()
        )

        if i % 100 == 0:
            print(f"  {method_name}: processed {i}/{len(texts)}")

    quality_row = {
        "Метод фильтрации": method_name,
        **metrics_from_predictions(y_true, predictions),
    }
    latency_row = {
        "Метод фильтрации": method_name,
        **latency_stats(latencies_ms),
    }

    predictions_df = df.copy()
    predictions_df["method"] = method_name
    predictions_df["prediction"] = predictions
    predictions_df["decision"] = decisions
    predictions_df["risk_score"] = risk_scores
    predictions_df["latency_ms"] = latencies_ms
    predictions_df["reason_codes"] = reason_codes
    predictions_df["reason_sources"] = reason_sources
    predictions_df["reason_actions"] = reason_actions
    predictions_df["ml_features"] = ml_features
    predictions_df["used_fallback"] = used_fallback

    return quality_row, latency_row, predictions_df


def evaluate_ablation(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    quality_rows = []
    latency_rows = []
    detail_frames = []

    for config in ABLATION_CONFIGS:
        print(f"\nEvaluating ablation: {config['name']}")
        q, l, d = evaluate_api_config(df, config)
        quality_rows.append(q)
        latency_rows.append(l)
        detail_frames.append(d)

    quality_df = pd.DataFrame(quality_rows)
    latency_df = pd.DataFrame(latency_rows)
    config_df = pd.DataFrame([
        {
            "Конфигурация фильтра": c["name"],
            "Rule detection": c["Rule detection"],
            "Heuristics": c["Heuristics"],
            "ML модель": c["ML модель"],
            "Rewrite": c["Rewrite"],
        }
        for c in ABLATION_CONFIGS
    ])

    ablation_df = (
        config_df
        .merge(
            quality_df.rename(columns={"Метод фильтрации": "Конфигурация фильтра"}),
            on="Конфигурация фильтра",
            how="left",
        )
        .merge(
            latency_df.rename(columns={"Метод фильтрации": "Конфигурация фильтра"}),
            on="Конфигурация фильтра",
            how="left",
        )
    )
    details_df = pd.concat(detail_frames, ignore_index=True)

    return ablation_df, details_df, quality_df


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Test dataset not found: {DATASET_PATH}")

    warmup_api()
    df = pd.read_csv(DATASET_PATH)

    print("\n" + "=" * 50)
    print("НАЧАЛО ГЕНЕРАЦИИ ТАБЛИЦ ДЛЯ ДИССЕРТАЦИИ")
    print("=" * 50)

    # 4.4.1 Test set composition
    composition_df = build_test_set_composition(df)
    composition_df.to_csv(
        RESULTS_DIR / "table_4_4_1_test_set_composition.csv",
        index=False,
    )
    print("\nТаблица 4.4.1 — Структура тестовой выборки:")
    print(composition_df.to_string(index=False))

    quality_rows = []
    latency_rows = []
    detail_frames = []

    q, l, d = evaluate_no_filter(df)
    quality_rows.append(q)
    latency_rows.append(l)
    detail_frames.append(d)

    for config in MAIN_CONFIGS:
        print(f"\nEvaluating: {config['name']}...")
        q, l, d = evaluate_api_config(df, config)
        quality_rows.append(q)
        latency_rows.append(l)
        detail_frames.append(d)

    quality_df = pd.DataFrame(quality_rows)
    latency_df = pd.DataFrame(latency_rows)
    all_predictions_df = pd.concat(detail_frames, ignore_index=True)

    all_predictions_df.to_csv(
        RESULTS_DIR / "all_predictions_detailed.csv",
        index=False,
    )

    # 4.4.2 Confusion Matrix
    confusion_df = quality_df[["Метод фильтрации", "TP", "TN", "FP", "FN"]]
    confusion_df.to_csv(
        RESULTS_DIR / "table_4_4_2_confusion_matrix.csv",
        index=False,
    )
    print("\nТаблица 4.4.2 — Матрица ошибок (Confusion Matrix):")
    print(confusion_df.to_string(index=False))

    # 4.4.3 Metrics
    metrics_df = quality_df[[
        "Метод фильтрации",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "FPR",
        "FNR",
    ]]
    metrics_df.to_csv(
        RESULTS_DIR / "table_4_4_3_quality_metrics.csv",
        index=False,
    )
    print("\nТаблица 4.4.3 — Метрики качества бинарного обнаружения:")
    print(metrics_df.to_string(index=False))

    # 4.4.4 Latency
    latency_df.to_csv(
        RESULTS_DIR / "table_4_4_4_latency.csv",
        index=False,
    )
    print("\nТаблица 4.4.4 — Анализ задержки (Latency):")
    print(latency_df.to_string(index=False))

    # 4.4.5 Categories
    final_module_df = all_predictions_df[all_predictions_df["method"] == FINAL_MODULE_NAME]
    if not final_module_df.empty:
        category_df = evaluate_module_by_categories(final_module_df)
        category_df.to_csv(
            RESULTS_DIR / "table_4_4_5_category_results.csv",
            index=False,
        )
        print("\nТаблица 4.4.5 — Результаты обнаружения по категориям (для финального модуля):")
        print(category_df.to_string(index=False))
    else:
        print(f"\n[Warning] Не найдены результаты для финального модуля: {FINAL_MODULE_NAME}")

    # 4.4.6 ML Model Comparison
    ml_comp_df = build_ml_comparison_table(quality_df, latency_df)
    ml_comp_df.to_csv(
        RESULTS_DIR / "table_4_4_6_ml_model_comparison.csv",
        index=False,
    )
    print("\nТаблица 4.4.6 — Сравнение моделей машинного обучения:")
    print(ml_comp_df.to_string(index=False))

    # 4.4.7 Ablation
    print("\nЗапуск Ablation Study (отключение компонентов)...")
    ablation_df, ablation_details_df, ablation_quality_df = evaluate_ablation(df)
    ablation_df.to_csv(
        RESULTS_DIR / "table_4_4_7_ablation_components.csv",
        index=False,
    )
    ablation_details_df.to_csv(
        RESULTS_DIR / "ablation_predictions_detailed.csv",
        index=False,
    )
    print("\nТаблица 4.4.7 — Влияние отдельных компонентов (Ablation Study):")
    print(ablation_df.to_string(index=False))

    # 4.4.8 Routing
    routing_df = build_routing_distribution(all_predictions_df)
    routing_df.to_csv(
        RESULTS_DIR / "table_4_4_8_routing_distribution.csv",
        index=False,
    )
    print("\nТаблица 4.4.8 — Распределение маршрутизации трафика (Allow/Rewrite/Block):")
    print(routing_df.to_string(index=False))

    # 4.4.9 Action metrics
    action_metrics_df = build_action_metrics(all_predictions_df)
    action_metrics_df.to_csv(
        RESULTS_DIR / "table_4_4_9_action_metrics.csv",
        index=False,
    )
    print("\nТаблица 4.4.9 — Метрики действий фильтра:")
    print(action_metrics_df.to_string(index=False))

    # 4.4.10 Category results for all methods
    category_all_df = evaluate_all_methods_by_categories(all_predictions_df)
    category_all_df.to_csv(
        RESULTS_DIR / "table_4_4_10_category_results_all_methods.csv",
        index=False,
    )
    print("\nТаблица 4.4.10 — Результаты по категориям для всех методов:")
    print(category_all_df.to_string(index=False))

    # 4.4.11 Delta vs baseline
    delta_df = build_delta_table(quality_df, baseline_name=BASELINE_NAME)
    delta_df.to_csv(
        RESULTS_DIR / "table_4_4_11_delta_vs_baseline.csv",
        index=False,
    )
    print("\nТаблица 4.4.11 — Изменение метрик относительно baseline:")
    print(delta_df.to_string(index=False))

    # 4.4.12 Fallback diagnostics
    fallback_df = all_predictions_df.groupby("method", as_index=False).agg(
        **{
            "Количество запросов": ("text", "count"),
            "Fallback использован, шт.": ("used_fallback", "sum"),
        }
    )
    fallback_df["Fallback использован, %"] = (
        fallback_df["Fallback использован, шт."]
        / fallback_df["Количество запросов"].clip(lower=1)
        * 100
    ).round(2)
    fallback_df.to_csv(
        RESULTS_DIR / "table_4_4_12_fallback_diagnostics.csv",
        index=False,
    )
    print("\nТаблица 4.4.12 — Диагностика fallback ML-моделей:")
    print(fallback_df.to_string(index=False))

    print("\n" + "=" * 50)
    print("ВСЕ ЭКСПЕРИМЕНТЫ УСПЕШНО ЗАВЕРШЕНЫ")
    print(f"Файлы *.csv сохранены в директории: {RESULTS_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
