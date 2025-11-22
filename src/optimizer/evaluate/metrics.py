from typing import List, Dict, Tuple, Optional


def build_placement(keys:       List[List[int]],
                    fingers:    List[List[int]],
                    effort:     List[List[int]]
                    ) -> Dict[int, Tuple[int, int, int, int]]:
    """
    Build mapping key -> (row, col) for O(1) lookups.
    Keys with value 0 (unused) will be omitted.
    """
    placement = {}
    for r, row in enumerate(keys):
        for c, k in enumerate(row):
            if k:
                placement[k] = (r, c, fingers[r][c], effort[r][c]/10)
    return placement


def get_sfb(bigrams: List[Tuple[int, int, int]],
            placement: Optional[Dict[int, Tuple[int, int, int, int]]]) -> int:
    score = 0.0
    for a, b, freq in bigrams:
        pa = placement.get(a)
        pb = placement.get(b)
        if not pa or not pb:
            continue
        r1, c1, f1, e1 = pa
        r2, c2, f2, e2 = pb
        if f1 == f2 and c1 == c2:
            score += freq * 0.1 + ((e1 + e2) / 2)
    return score
