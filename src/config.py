from utils import combine_matrices, get_unigram, get_bigram, get_trigram


KEYMAP = {
    'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5,
    'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10,
    'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15,
    'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20,
    'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25,
    'z': 26
}

l_layout_template = [
    [None, 0, 0, 0, None],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
]

r_layout_template = [
    [None, 0, 0, 0, None],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
]

l_finger_map = [
    [0, 1, 2, 3, 3],
    [0, 1, 2, 3, 3],
    [0, 1, 2, 3, 3]
]

r_finger_map = [
    [4, 4, 5, 6, 7],
    [4, 4, 5, 6, 7],
    [4, 4, 5, 6, 7]
]

l_effort_map = [
    [8, 3, 3, 3, 7],
    [2, 0, 0, 0, 6],
    [9, 6, 6, 6, 8],
]

r_effort_map = [
    [7, 3, 3, 3, 8],
    [6, 0, 0, 0, 2],
    [8, 6, 6, 6, 9],
]

l_distance_map = [
    [0.18,  0.18,    0.18,    0.18,    0.23],
    [0.00,  0.00,    0.00,    0.00,    0.20],
    [0.18,  0.18,    0.18,    0.18,    0.30],
]

r_distance_map = [
    [0.23,  0.18,    0.18,    0.18,    0.18],
    [0.20,  0.00,    0.00,    0.00,    0.00],
    [0.30,  0.18,    0.18,    0.18,    0.18],
]

LAYOUT_TEMPLATE = combine_matrices(l_layout_template, r_layout_template)
FINGER_MAP = combine_matrices(l_finger_map, r_finger_map)
EFFORT_MAP = combine_matrices(l_effort_map, r_effort_map)
DISTANCE_MAP = combine_matrices(l_distance_map, r_distance_map)
UNIGRAMS = get_unigram("data/english_1grams.csv", KEYMAP)
BIGRAMS = get_bigram("data/english_2grams.csv", KEYMAP)
TRIGRAMS = get_trigram("data/english_3grams.csv", KEYMAP)
