from typing import List, Dict, Tuple, Optional
from config import (LAYOUT_TEMPLATE, FINGER_MAP, EFFORT_MAP,
                    DISTANCE_MAP, UNIGRAMS, BIGRAMS, TRIGRAMS)
from optimizer.evaluate.metrics import build_placement, get_sfb, get_effort
from utils import apply_layout


def evaluate(
    key_sequence: list,
) -> Tuple[float, Dict]:
    layout = apply_layout(key_sequence, LAYOUT_TEMPLATE)
    placement = build_placement(layout, FINGER_MAP, EFFORT_MAP)
    sfb = get_sfb(BIGRAMS, placement)
    effort = get_effort(UNIGRAMS, placement)

    score = (sfb*10)+effort
    return None, score
