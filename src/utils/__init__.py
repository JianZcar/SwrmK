import csv
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


def combine_matrices(left, right):
    return [l_row + r_row for l_row, r_row in zip(left, right)]


def apply_layout(s_char, layout_template):
    """
    s_char: 1D algorithm output list of ints (0 means skip)
    keys: 2D keyboard template (0 = available, None = unusable)
    """
    mapped = [row[:] for row in layout_template]  # deep copy
    i = 0
    for row in range(len(layout_template)):
        for col in range(len(layout_template[0])):
            if mapped[row][col] is not None:
                mapped[row][col] = s_char[i]
                i += 1

    clean = [[0 if v is None else int(v) for v in row] for row in mapped]
    return clean


def layout_to_letters(num_layout, keymap):
    """
    Converts a numeric keyboard layout to lettered layout using keymap.

    num_layout: 2D list of ints (output from apply_layout)
    keymap: dict mapping letters to numbers (e.g., 'a': 1)
    """
    # Create reverse mapping for easy lookup
    rev_map = {v: k for k, v in keymap.items()}

    letter_layout = []
    for row in num_layout:
        letter_row = []
        for v in row:
            if v == 0:
                letter_row.append('')  # empty slot
            else:
                letter_row.append(rev_map.get(v, '?'))
        letter_layout.append(letter_row)

    return letter_layout


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


def get_unigrams(path, definition):
    letter_freq = {}
    rows = []
    total_freq = 0
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            total_freq += int(row['freq'])

    for row in rows:
        letter = row['unigram'].lower()
        count = int(row['freq'])
        k = definition.get(letter)
        if k:
            letter_freq[k] = count / total_freq
    return letter_freq


def get_bigrams(path, definition):
    bigrams = []
    rows = []
    total_freq = 0
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            total_freq += int(row['freq'])

    for row in rows:
        bi = row.get('bigram', '')
        if not bi or len(bi) != 2:
            continue  # skip empty or malformed entries
        freq = int(row['freq'])
        ka = definition.get(bi[0].lower())
        kb = definition.get(bi[1].lower())
        if ka and kb:
            bigrams.append((ka, kb, freq / total_freq))
    return bigrams


def get_trigrams(path, definition):
    trigrams = []
    rows = []
    total_freq = 0
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            total_freq += int(row['freq'])

    for row in rows:
        tri = row.get('trigram', '')
        if not tri or len(tri) != 3:
            continue
        freq = int(row['freq'])
        ka = definition.get(tri[0].lower())
        kb = definition.get(tri[1].lower())
        kc = definition.get(tri[2].lower())
        if ka and kb and kc:
            trigrams.append((ka, kb, kc, freq / total_freq))
    return trigrams


def get_skip_bigrams(path, definition, skip=1):
    totals = defaultdict(float)
    total_freq = 0
    rows = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row["word"]
            freq = int(row["freq"])
            rows.append((word, freq))
            total_freq += freq

    for word, freq in rows:
        word = word.strip()
        if len(word) <= skip:
            continue

        for i in range(len(word) - skip - 1):
            c1 = word[i].lower()
            c2 = word[i + skip + 1].lower()
            if c1 == c2:
                continue

            k1 = definition.get(c1)
            k2 = definition.get(c2)
            if k1 is None or k2 is None:
                continue

            rel_freq = freq / total_freq
            totals[(k1, k2)] += rel_freq

    skip_bigrams = [(k1, k2, freq_sum)
                    for (k1, k2), freq_sum in totals.items()]
    return skip_bigrams
