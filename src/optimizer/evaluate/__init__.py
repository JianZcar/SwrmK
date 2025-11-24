from typing import List, Dict, Tuple, Optional
from config import (WEIGHTS, LAYOUT_TEMPLATE, FINGER_MAP, EFFORT_MAP,
                    DISTANCE_MAP, UNIGRAMS, BIGRAMS, TRIGRAMS, SKIP_BIGRAMS)
from optimizer.evaluate.metrics import (get_sfb_score, get_uge_score,
                                        get_lsb_score, get_scissors_score)
from utils import apply_layout, build_placement


def evaluate(
    s_char: list,
    get_metrics: bool
) -> Tuple[float, Dict]:
    layout = apply_layout(s_char, LAYOUT_TEMPLATE)
    placement = build_placement(layout, FINGER_MAP, EFFORT_MAP)

    sfb = get_sfb_score(BIGRAMS, placement)
    uge = get_uge_score(UNIGRAMS, placement)
    lsb = get_lsb_score(BIGRAMS, placement, layout)
    sfsb = get_sfb_score(SKIP_BIGRAMS, placement)
    scissors = get_scissors_score(BIGRAMS, placement)

    metrics = {
        "SFB": sfb,
        "UGE": uge,
        "LSB": lsb,
        "SFSB": sfsb,
        "SCISSORS": scissors
    }

    weights = WEIGHTS

    if get_metrics:
        return metrics

    return sum(metrics[k] * weights[k] for k in metrics)
