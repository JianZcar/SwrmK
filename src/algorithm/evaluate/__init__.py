from typing import List, Dict, Tuple, Optional
from algorithm.evaluate.metrics import metrics_matrix


def evaluate(
    keys: List[List[int]],
    fingers: List[List[int]],
    effort: List[List[float]],
    unigrams: Dict[int, int],
    bigrams: List[Tuple[int, int, int]],
    trigrams: Optional[List[Tuple[int, int, int, int]]] = None,
    distance_matrix: Optional[List[List[float]]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict]:
    """
    Evaluate combined matrices and return (score, metrics).

    Inputs are assumed COMBINED (left+right already merged row-wise).
    - letter_freq: {key_num: count}
    - bigrams: list of (a_key, b_key, count)
    - trigrams: optional list of (a,b,c,count)
    - distance_matrix: optional combined distance matrix (same shape as keys)
    - weights: optional dict overriding defaults
    """

    # 1) compute metrics (metrics_matrix builds key_pos internally)
    metrics = metrics_matrix(
        keys=keys,
        fingers=fingers,
        effort=effort,
        unigrams=unigrams or {},
        bigrams=bigrams or [],
        trigrams=trigrams,
        distance_matrix=distance_matrix
    )

    # 2) default weights (tweak these to taste)
    default_weights: Dict[str, float] = {
        "sfb": 1000.0,
        "effort": 100.0,
        "scissors": 1.0,
        "prscissors": 1.0,
        "wide_scissors": 1.0,
        "hand_balance": 10.0,
        "finger_load": 100.0,  # Handled specially below
        "distance": 20.0,
        "hand_alternation": 1.0,
        "rolls": 50.0,  # Handled specially below
    }

    # Merge user weights (if provided)
    if weights:
        default_weights.update(weights)

    # 3) Combine metric values into a single float score.
    #    For scalar metrics, multiply directly.
    #    For dict-valued metrics (finger_load, rolls) we handle them explicitly:
    score = 0.0

    # Scalar metrics (present in metrics as numbers)
    scalar_keys = [
        "sfb", "effort", "scissors", "prscissors", "wide_scissors",
        "hand_balance", "distance", "hand_alternation"
    ]
    for k in scalar_keys:
        val = metrics.get(k, 0)
        w = default_weights.get(k, 0.0)
        # defensive: ensure numeric
        try:
            score += float(val) * float(w)
        except Exception:
            # if val can't be cast to float, skip (or log)
            continue

    # Optional: include finger_load as a penalty for overloaded fingers
    # default_weights["finger_load"] tells per-finger weight (single scalar) or 0 to skip.
    fl_w = default_weights.get("finger_load", 0.0)
    if fl_w != 0.0 and isinstance(metrics.get("finger_load"), dict):
        # Simple strategy: penalize variance or max usage. Example uses max*weight.
        finger_load = metrics["finger_load"]
        max_load = max(finger_load.values()) if finger_load else 0
        score += max_load * fl_w

    # Optional: include rolls (inward/outward)
    rolls_w = default_weights.get("rolls", 0.0)
    if rolls_w != 0.0 and isinstance(metrics.get("rolls"), dict):
        rolls = metrics["rolls"]
        # Example: treat outward rolls as penalty, inward as smaller penalty (customize as needed)
        outward = rolls.get("outward", 0)
        # inward = rolls.get("inward", 0)
        score += outward * rolls_w
        # if you want different scaling: score += inward * (rolls_w * 0.5)

    # 4) Return final score and full metric breakdown
    # if metrics["sfb"] <= 6:
    #     print(metrics)
    return float(score)
