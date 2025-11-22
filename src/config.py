from utils import combine_matrices, get_unigram, get_bigram, get_trigram

DEFAULT_PINKIES = {0, 9}    # example: first and last fingers
DEFAULT_RINGS = {1, 8}      # example: second and second-last fingers

KEYMAP = {
    'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5,
    'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10,
    'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15,
    'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20,
    'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25,
    'z': 26
}

keys_left = [
    [None, 0, 0, 0, None],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
]

keys_right = [
    [None, 0, 0, 0, None],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
]

fingers_left = [
    [0, 1, 2, 3, 4],
    [0, 1, 2, 3, 4],
    [0, 1, 2, 3, 4]
]

fingers_right = [
    [5, 5, 6, 7, 8],
    [5, 5, 6, 7, 8],
    [5, 5, 6, 7, 8]
]

effort_left = [
    [8, 3, 3, 3, 7],
    [2, 0, 0, 0, 6],
    [9, 7, 7, 7, 8],
]

effort_right = [
    [7, 2, 2, 2, 8],
    [6, 0, 0, 0, 2],
    [8, 5, 5, 5, 9],
]

distance_left = [
    [0.18,  0.18,    0.18,    0.18,    0.23],
    [0.00,  0.00,    0.00,    0.00,    0.20],
    [0.18,  0.18,    0.18,    0.18,    0.30],
]

distance_right = [
    [0.23,  0.18,    0.18,    0.18,    0.18],
    [0.20,  0.00,    0.00,    0.00,    0.00],
    [0.30,  0.18,    0.18,    0.18,    0.18],
]

KEYS = combine_matrices(keys_left, keys_right)
FINGERS = combine_matrices(fingers_left, fingers_right)
EFFORT = combine_matrices(effort_left, effort_right)
DISTANCE = combine_matrices(distance_left, distance_right)
UNIGRAMS = get_unigram("data/english_1grams.csv", KEYMAP)
BIGRAMS = get_bigram("data/english_2grams.csv", KEYMAP)
TRIGRAMS = get_trigram("data/english_3grams.csv", KEYMAP)
