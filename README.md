# EvoKana

EvoKana is a Python-based framework designed for the autonomous **evolution and optimization of keyboard layouts**. Inspired by the systematic arrangement and learning inherent in 'kana' syllabaries, this project employs advanced computational intelligence techniques to discover highly efficient and ergonomic keyboard configurations.

It works but it needs more refinement

## What it Does

The core purpose of EvoKana is to mathematically evaluate and iteratively refine keyboard layouts to minimize typing effort and maximize efficiency. It simulates the natural "evolution" of character placement on a keyboard, adapting to statistical language patterns and ergonomic considerations.

### Key Features:

*   **Keyboard Layout Optimization:** Utilizes a **Memetic Algorithm combined with Simulated Annealing** to search for optimal keyboard layouts. This hybrid approach leverages the global search capabilities of evolutionary algorithms with the local refinement of simulated annealing.
*   **Comprehensive Metric Evaluation:** Each potential keyboard layout is rigorously assessed using a suite of metrics defined in `src/algorithm/evaluate/metrics.py` and `src/algorithm/evaluate/__init__.py`. These metrics include:
    *   **Same Finger Bigrams (SFB):** Penalizes consecutive key presses on the same finger.
    *   **Typing Effort:** Calculates the physical effort required for keystrokes based on finger and key positions.
    *   **Finger Load:** Distributes typing load across fingers to prevent overuse of specific digits.
    *   **Hand Balance & Alternation:** Promotes balanced usage between left and right hands, and encourages hand alternation for rhythm.
    *   **Distance:** Measures the travel distance between consecutive keys.
    *   **Scissors & Rolls:** Identifies awkward finger movements (like "scissors" motions) and analyzes rolling patterns (inward/outward) within a hand.
*   **Statistical Language Modeling:** Integrates real-world language data to inform the optimization process:
    *   **Unigram Frequencies:** Uses single character frequencies (from `data/english_1grams.csv`).
    *   **Bigram Frequencies:** Accounts for common two-character sequences (from `data/english_2grams.csv`).
    *   **Trigram Frequencies:** Considers common three-character sequences (from `data/english_3grams.csv`).
*   **Configurable Keyboard Models:** Allows for defining custom keyboard physical properties such as key positions, finger assignments, and base effort values in `src/config.py`.

## How it Works

The `src/main.py` script orchestrates the optimization process. It initializes a population of random keyboard layouts, then evolves them over generations. In each generation, layouts are selected, recombined (crossover), mutated, and optionally refined using Simulated Annealing (local search) before being evaluated. The best-performing layouts, as determined by the `evaluate` function, are propagated, leading to progressively better keyboard designs.

The utility functions in `src/utils/__init__.py` handle tasks such as combining left and right-hand key matrices, applying the generated layouts to a keyboard template, converting numeric layouts to letter-based representations, and loading language frequency data.

## Usage

To run the keyboard layout optimization, execute `main.py`:

```bash
python src/main.py
```

The script will output the best layout found (represented as characters on the keyboard grid) and its corresponding score and detailed metric breakdown.

## Configuration

The `src/config.py` file contains all the parameters defining the keyboard's physical characteristics (e.g., `KEYS`, `FINGERS`, `EFFORT`, `DISTANCE`) and links to the language frequency data files. Users can modify these parameters to adapt the optimization to different keyboard types or linguistic datasets.
