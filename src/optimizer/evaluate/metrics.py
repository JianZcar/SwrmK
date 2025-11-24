from typing import List, Dict, Tuple, Optional


def get_sfb_score(bigrams: List[Tuple[int, int, int]],
                  placement: Optional[Dict[int, Tuple[int, int, int, int]]]
                  ) -> float:
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


def get_uge_score(unigrams: Dict[int, float],
                  placement: Optional[Dict[int, Tuple[int, int, int, int]]]
                  ) -> float:
    # Unigram Effort
    score = 0.0
    for char, freq in unigrams.items():
        p = placement[char]
        if not p:
            continue
        _r, _c, _f, e = p

        score += e * freq
    total_score = score * 100
    return total_score


def get_lsb_score(bigrams: List[Tuple[int, int, int]],
                  placement: Optional[Dict[int, Tuple[int, int, int, int]]],
                  layout: list
                  ) -> float:
    total_score = 0.0
    total_frequency = 0.0
    total_effort = 0.0
    len_half = len(layout[0])//2
    center_col = (len_half, len_half-1)

    for a, b, freq in bigrams:
        pa = placement.get(a)
        pb = placement.get(b)
        if not pa or not pb:
            continue
        _r1, c1, _f1, e1 = pa
        _r2, c2, _f2, e2 = pb

        for i, c3 in enumerate(center_col):
            c4 = c3 + (2-(4*i))
            if (c1 == c3 and c2 == c4) or (c2 == c3 and c1 == c4):
                total_frequency += freq
                total_effort += ((e1 + e2) / 2) * freq
                break

    frequency_score = total_frequency * 100
    effort_score = total_effort * 100
    total_score = frequency_score + effort_score
    return total_score


def get_scissors_score(bigrams: List[Tuple[int, int, int]],
                       placement: Optional[Dict[int,
                                                Tuple[int, int, int, int]]]
                       ) -> float:
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

        r_diff = abs(r1-r2)
        c_diff = abs(c1-c2)

        if r_diff == 2 and c_diff == 1 and f1 != f2:
            total_frequency += freq
            total_effort += ((e1 + e2) / 2) * freq

    frequency_score = total_frequency * 100
    effort_score = total_effort * 100
    total_score = frequency_score + effort_score
    return total_score
