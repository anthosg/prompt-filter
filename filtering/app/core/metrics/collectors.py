from prometheus_client import Counter, Histogram

FILTER_REQUESTS = Counter(
    "pf_filter_requests_total",
    "Total filtering requests",
    ["endpoint"],
)

FILTER_LATENCY = Histogram(
    "pf_filter_latency_seconds",
    "Filtering request latency in seconds",
    ["endpoint"],
)

FILTER_DECISIONS = Counter(
    "pf_filter_decisions_total",
    "Total filter decisions",
    ["decision"],
)
