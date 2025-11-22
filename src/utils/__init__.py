import csv


def combine_matrices(left, right):
    return [l_row + r_row for l_row, r_row in zip(left, right)]


def apply_layout(seq, keys):
    """
    seq: 1D algorithm output list of ints (0 means skip)
    keys: 2D keyboard template (0 = available, None = unusable)
    """
    mapped = [row[:] for row in keys]  # deep copy
    i = 0
    for row in range(len(keys)):
        for col in range(len(keys[0])):
            if mapped[row][col] is not None:
                mapped[row][col] = seq[i]
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
                # fallback to '?' if not found
                letter_row.append(rev_map.get(v, '?'))
        letter_layout.append(letter_row)

    return letter_layout


def get_unigram(path, definition):
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


def get_bigram(path, definition):
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


def get_trigram(path, definition):
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
