from typing import List, Dict, Tuple, Optional
from config import KEYS, FINGERS, EFFORT, DISTANCE, UNIGRAMS, BIGRAMS, TRIGRAMS
from optimizer.evaluate.metrics import build_placement, get_sfb
from utils import apply_layout


def evaluate(
    keys: list,
) -> Tuple[float, Dict]:
    keys = apply_layout(keys, KEYS)
    placement = build_placement(keys, FINGERS, EFFORT)
    score = get_sfb(BIGRAMS, placement)
    return None, score
