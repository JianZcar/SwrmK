from typing import List, Dict, Tuple, Optional


def build_placement(layout:     List[List[int]],
                    finger_map: List[List[int]],
                    effort:     List[List[int]]
                    ) -> Dict[int, Tuple[int, int, int, int]]:
    """
    Build mapping key -> (row, col) for O(1) lookups.
    Keys with value 0 (unused) will be omitted.
    """
    placement = {}
    for r, row in enumerate(layout):
        for c, k in enumerate(row):
            if k:
                placement[k] = (r, c, finger_map[r][c], effort[r][c]/10)
    return placement


def get_sfb(bigrams: List[Tuple[int, int, int]],
            placement: Optional[Dict[int, Tuple[int, int, int, int]]]) -> int:
    total_score = 0.0
    total_frequency = 0.0
    total_effort = 0.0
    for a, b, freq in bigrams:
        pa = placement.get(a)
        pb = placement.get(b)
        if not pa or not pb:
            continue
        r1, c1, f1, e1 = pa
        r2, c2, f2, e2 = pb

        if f1 == f2:
            total_frequency += freq
            total_effort += ((e1 + e2) / 2) * freq

    frequency_score = total_frequency * 100
    effort_score = total_effort * 100
    total_score = frequency_score + effort_score
    return total_score


def get_effort(unigrams: Dict[int, float],
               placement: Optional[Dict[int, Tuple[int, int, int, int]]]
               ):
    total_score = 0.0
    for char, freq in unigrams.items():
        p = placement[char]
        if not p:
            continue
        _r, _c, _f, e = p

        total_score += e * freq
    return total_score * 100
