import math
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


# ---------- Helpers ----------
def build_key_pos(keys: List[List[int]]) -> Dict[int, Tuple[int, int]]:
    """
    Build mapping key -> (row, col) for O(1) lookups.
    Keys with value 0 (unused) will be omitted.
    """
    pos = {}
    for r, row in enumerate(keys):
        for c, k in enumerate(row):
            if k:  # skip zero/empty
                pos[k] = (r, c)
    return pos


def get_hand_from_finger(finger: int, split_at: int = 4) -> str:
    return "L" if finger <= split_at else "R"


def sfb_metric(keys: List[List[int]],
               fingers: List[List[int]],
               bigrams: List[Tuple[int, int, int]],
               key_pos: Optional[Dict[int, Tuple[int, int]]] = None) -> int:
    if key_pos is None:
        key_pos = build_key_pos(keys)
    total = 0
    for a, b, count in bigrams:
        pa = key_pos.get(a)
        pb = key_pos.get(b)
        if not pa or not pb:
            continue
        r1, c1 = pa
        r2, c2 = pb
        if fingers[r1][c1] == fingers[r2][c2]:
            total += count
    return total * 100


def effort_metric(keys: List[List[int]],
                  fingers: List[List[int]],
                  effort: List[List[float]],
                  bigrams: List[Tuple[int, int, int]],
                  key_pos: Optional[Dict[int, Tuple[int, int]]] = None) -> float:
    """Return average effort per bigram (makes scale comparable)."""
    if key_pos is None:
        key_pos = build_key_pos(keys)
    total = 0.0
    total_count = 0
    for a, b, count in bigrams:
        pa = key_pos.get(a)
        pb = key_pos.get(b)
        if not pa or not pb:
            continue
        r1, c1 = pa
        r2, c2 = pb
        total += (((effort[r1][c1] + effort[r2][c2]) / 2.0)**2) * count
        total_count += count
    if total_count == 0:
        return 0.0
    return total / total_count


def finger_load_metric(keys: List[List[int]],
                       fingers: List[List[int]],
                       letter_freq: Dict[int, int],
                       key_pos: Optional[Dict[int, Tuple[int, int]]] = None) -> Dict[int, int]:
    if key_pos is None:
        key_pos = build_key_pos(keys)
    load = defaultdict(int)
    for k, cnt in letter_freq.items():
        pos = key_pos.get(k)
        if not pos:
            continue
        r, c = pos
        load[fingers[r][c]] += cnt
    # convert to normal dict (and ensure keys 1..10 exist)
    return {f: load.get(f, 0) * 100 for f in range(1, 11)}





def distance_metric(keys: List[List[int]],
                    bigrams: List[Tuple[int, int, int]],
                    key_pos: Optional[Dict[int, Tuple[int, int]]] = None,
                    scale_x: float = 1.0,
                    scale_y: float = 1.0) -> float:
    """
    Return average (per-bigram) Euclidean distance to keep scale similar to effort.
    """
    if key_pos is None:
        key_pos = build_key_pos(keys)
    total = 0.0
    total_count = 0
    for a, b, count in bigrams:
        pa = key_pos.get(a)
        pb = key_pos.get(b)
        if not pa or not pb:
            continue
        r1, c1 = pa
        r2, c2 = pb
        dx = (c1 - c2) * scale_x
        dy = (r1 - r2) * scale_y
        total += math.hypot(dx, dy) * count
        total_count += count
    if total_count == 0:
        return 0.0
    return total / total_count


def distance_using_matrix(distance_matrix: List[List[float]],
                          keys: List[List[int]],
                          bigrams: List[Tuple[int, int, int]],
                          key_pos: Optional[Dict[int, Tuple[int, int]]] = None) -> float:
    """
    If you already have a per-key distance matrix (same shape as keys), this function
    will use the *difference* between the two per-key values as the 'distance' proxy.
    (There's no single canonical interpretation — adjust as needed.)
    """
    if key_pos is None:
        key_pos = build_key_pos(keys)
    total = 0.0
    for a, b, count in bigrams:
        pa = key_pos.get(a)
        pb = key_pos.get(b)
        if not pa or not pb:
            continue
        r1, c1 = pa
        r2, c2 = pb
        total += abs(distance_matrix[r1][c1] - distance_matrix[r2][c2]) * count
    return total


def hand_balance_metric(keys, fingers,
                        letter_freq: Dict[int, int], key_pos=None):
    if key_pos is None:
        key_pos = build_key_pos(keys)
    left = 0
    right = 0
    total = sum(letter_freq.values())
    for k, cnt in letter_freq.items():
        pos = key_pos.get(k)
        if not pos:
            continue
        r, c = pos
        if get_hand_from_finger(fingers[r][c]) == "L":
            left += cnt
        else:
            right += cnt
    if total == 0:
        return 0.0
    return abs(left - right) / total * 100.0


def hand_alternation_metric(keys, fingers, bigrams, key_pos=None) -> float:
    if key_pos is None:
        key_pos = build_key_pos(keys)
    if not bigrams:
        return 0.0
    alternated = 0
    total = 0
    for a, b, count in bigrams:
        pa = key_pos.get(a)
        pb = key_pos.get(b)
        if not pa or not pb:
            continue
        r1, c1 = pa
        r2, c2 = pb
        total += count
        if get_hand_from_finger(fingers[r1][c1]) != get_hand_from_finger(fingers[r2][c2]):
            alternated += count
    return (alternated / total) * 100.0 if total else 0.0


def scissors_metric(keys, fingers, bigrams, key_pos=None):
    if key_pos is None:
        key_pos = build_key_pos(keys)
    total = 0
    for a, b, count in bigrams:
        pa = key_pos.get(a)
        pb = key_pos.get(b)
        if not pa or not pb:
            continue
        r1, c1 = pa
        r2, c2 = pb
        fa, fb = fingers[r1][c1], fingers[r2][c2]
        if get_hand_from_finger(fa) != get_hand_from_finger(fb):
            continue
        if abs(r1 - r2) == 2 and abs(fa - fb) == 1:
            total += count
    return total * 100


def prscissors_metric(keys, fingers, bigrams, rings=None, pinkies=None, key_pos=None):
    # These were originally imported from config, now passed or fallback to default
    from config import DEFAULT_RINGS, DEFAULT_PINKIES
    rings = rings if rings is not None else DEFAULT_RINGS
    pinkies = pinkies if pinkies is not None else DEFAULT_PINKIES

    if key_pos is None:
        key_pos = build_key_pos(keys)
    total = 0
    for a, b, count in bigrams:
        pa = key_pos.get(a)
        pb = key_pos.get(b)
        if not pa or not pb:
            continue
        r1, c1 = pa
        r2, c2 = pb
        fa, fb = fingers[r1][c1], fingers[r2][c2]
        if (fa in rings and fb in pinkies) or (fb in rings and fa in pinkies):
            if abs(r1 - r2) >= 1:
                total += count
    return total * 100


def wide_scissors_metric(keys, fingers, bigrams, key_pos=None):
    if key_pos is None:
        key_pos = build_key_pos(keys)
    total = 0
    for a, b, count in bigrams:
        pa = key_pos.get(a)
        pb = key_pos.get(b)
        if not pa or not pb:
            continue
        r1, c1 = pa
        r2, c2 = pb
        fa, fb = fingers[r1][c1], fingers[r2][c2]
        if get_hand_from_finger(fa) != get_hand_from_finger(fb):
            continue
        if abs(r1 - r2) == 2 and abs(c1 - c2) > 1:
            total += count
    return total * 100


def rolls_metric(keys, fingers, bigrams, key_pos=None) -> Dict[str, int]:
    if key_pos is None:
        key_pos = build_key_pos(keys)
    inward = 0
    outward = 0
    for a, b, count in bigrams:
        pa = key_pos.get(a)
        pb = key_pos.get(b)
        if not pa or not pb:
            continue
        r1, c1 = pa
        r2, c2 = pb
        fa, fb = fingers[r1][c1], fingers[r2][c2]
        if get_hand_from_finger(fa) != get_hand_from_finger(fb):
            continue
        if fa < fb:
            inward += count
        elif fa > fb:
            outward += count
    return {"inward": inward * 100, "outward": outward * 100}


# ---------- Wrapper that returns all metrics ----------
def metrics_matrix(keys: List[List[int]],
                   fingers: List[List[int]],
                   effort: List[List[float]],
                   unigrams: Dict[int, int],
                   bigrams: List[Tuple[int, int, int]],
                   trigrams: Optional[List[Tuple[int, int, int, int]]] = None,
                   distance_matrix: Optional[List[List[float]]] = None) -> Dict:
    key_pos = build_key_pos(keys)
    out = {
        "sfb": sfb_metric(keys, fingers, bigrams, key_pos),
        "effort": effort_metric(keys, fingers, effort, bigrams, key_pos),
        "hand_balance": hand_balance_metric(keys, fingers, unigrams, key_pos),
        "finger_load": finger_load_metric(keys, fingers, unigrams, key_pos),

        "distance": (distance_using_matrix(distance_matrix, keys, bigrams, key_pos)
                     if distance_matrix is not None
                     else distance_metric(keys, bigrams, key_pos)),
        "hand_alternation": hand_alternation_metric(keys, fingers, bigrams, key_pos),
        "scissors": scissors_metric(keys, fingers, bigrams, key_pos),
        "prscissors": prscissors_metric(keys, fingers, bigrams, key_pos=key_pos),
        "wide_scissors": wide_scissors_metric(keys, fingers, bigrams, key_pos),
        "rolls": rolls_metric(keys, fingers, bigrams, key_pos),
    }
    return out
