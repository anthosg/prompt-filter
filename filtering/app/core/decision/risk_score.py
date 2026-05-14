def combine_scores(rule_score: float, ml_score: float) -> float:
    score = rule_score + ml_score - (rule_score * ml_score)
    return max(0.0, min(score, 1.0))