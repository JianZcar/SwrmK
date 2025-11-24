from typing import List, Dict, Tuple, Optional
from config import (WEIGHTS, LAYOUT_TEMPLATE, FINGER_MAP, EFFORT_MAP,
                    DISTANCE_MAP, UNIGRAMS, BIGRAMS, TRIGRAMS)
from optimizer.evaluate.metrics import build_placement, get_sfb, get_uge
from utils import apply_layout


def evaluate(
    key_sequence: list,
    get_metrics: bool
) -> Tuple[float, Dict]:
    layout = apply_layout(key_sequence, LAYOUT_TEMPLATE)
    placement = build_placement(layout, FINGER_MAP, EFFORT_MAP)

    sfb = get_sfb(BIGRAMS, placement)
    uge = get_uge(UNIGRAMS, placement)

    metrics = {
        "SFB": sfb,
        "UGE": uge
    }

    weights = WEIGHTS

    if get_metrics:
        return metrics

    return sum(metrics[k] * weights[k] for k in metrics)
