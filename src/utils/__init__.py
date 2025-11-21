def combine_matrices(left, right):
    return [l_row + r_row for l_row, r_row in zip(left, right)]


def apply_layout(seq, keys):
    """
    seq: 1D algorithm output list of ints (0 means skip)
    keys: 2D keyboard template (0 = available, None = unusable)
    """
    mapped = [row[:] for row in keys]  # deep copy

    # collect available positions (only the 0 slots)
    slots = [(r, c)
             for r, row in enumerate(keys)
             for c, v in enumerate(row)
             if v == 0]

    slot_i = 0  # index into slots

    for v in seq:
        if v == 0:
            # skip algorithm empty; DO NOT consume a 0 slot
            continue

        if slot_i >= len(slots):
            break  # no more available positions

        r, c = slots[slot_i]
        mapped[r][c] = v
        slot_i += 1

    clean = [[0 if v is None else int(v) for v in row] for row in mapped]
    return clean
