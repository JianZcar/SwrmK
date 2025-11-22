import os
import numpy as np
from tabulate import tabulate
from concurrent.futures import ProcessPoolExecutor
from utils import apply_layout, layout_to_letters
from config import (KEYS, FINGERS, EFFORT, DISTANCE,
                    UNIGRAMS, BIGRAMS, TRIGRAMS, KEYMAP)

BEST_LAYOUT = []


# ---------------------------
# Helpers (top-level, picklable)
# ---------------------------

def evaluate_layout(layout, eval_fn):
    """Apply layout (slot->letter array) to KEYS and evaluate with eval_fn."""
    matrix_layout = apply_layout(layout, KEYS)
    score = float(eval_fn(matrix_layout, FINGERS, EFFORT, UNIGRAMS,
                          BIGRAMS, TRIGRAMS, DISTANCE)) if eval_fn else np.inf
    return score


def random_layout(n_slots, n_letters, rng):
    """Create a random layout array (slot->letter values 0..n_letters)."""
    slots = np.arange(n_slots)
    rng.shuffle(slots)
    layout = np.zeros(n_slots, dtype=int)
    for letter_idx, slot in enumerate(slots[:n_letters], start=1):
        layout[slot] = letter_idx
    return layout


def population_init(pop_size, n_slots, n_letters, rng):
    """Initialize population of slot->letter layouts."""
    pop = np.zeros((pop_size, n_slots), dtype=int)
    for i in range(pop_size):
        pop[i] = random_layout(n_slots, n_letters, rng)
    return pop


def tournament_select(pop, scores, rng, k=3):
    """Tournament selection: return a copy of the winner layout."""
    idxs = rng.integers(0, pop.shape[0], size=k)
    winner = idxs[np.argmin(scores[idxs])]
    return pop[winner].copy()


def segment_crossover(parent_a, parent_b, rng):
    n = parent_a.shape[0]
    a, b = parent_a, parent_b
    child = np.zeros(n, dtype=int)
    i, j = sorted(rng.integers(0, n, size=2))
    child[i:j+1] = a[i:j+1]
    used = set(child[child != 0].tolist())
    b_vals = [v for v in b.tolist() if v != 0 and v not in used]
    b_iter = iter(b_vals)
    for pos in range(n):
        if child[pos] == 0:
            try:
                child[pos] = next(b_iter)
            except StopIteration:
                break
    missing = [x for x in range(
        1, max(1, int(max(a.max(), b.max())) + 1)) if x not in set(child.tolist())]
    if missing:
        miss_iter = iter(missing)
        for pos in range(n):
            if child[pos] == 0:
                try:
                    child[pos] = next(miss_iter)
                except StopIteration:
                    break
    return child


def swap_mutation(layout, rng, swap_prob=0.1):
    new = layout.copy()
    n = len(new)
    attempts = max(1, int(np.ceil(n * swap_prob)))
    for _ in range(attempts):
        i, j = rng.integers(0, n, size=2)
        new[i], new[j] = new[j], new[i]
    return new


# ---------------------------
# Simulated Annealing (no printing; worker-safe)
# ---------------------------

def simulated_annealing_local_search_worker(init_layout, eval_fn,
                                            iters=200, t0=1.0, alpha=0.95,
                                            seed=None):
    """
    SA worker function -- NO prints. Returns (best_layout, best_score).
    Uses its own RNG seeded from OS if seed is None.
    """
    rng = np.random.default_rng(seed)
    current = init_layout.copy()
    best = current.copy()
    best_score = evaluate_layout(best, eval_fn)
    current_score = best_score
    T = t0
    n_slots = len(current)

    for _ in range(iters):
        i, j = rng.integers(0, n_slots, size=2)
        neighbor = current.copy()
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        neigh_score = evaluate_layout(neighbor, eval_fn)
        delta = neigh_score - current_score
        if delta < 0 or rng.random() < np.exp(-delta / max(1e-12, T)):
            current, current_score = neighbor, neigh_score
            if current_score < best_score:
                best, best_score = current.copy(), current_score
        T *= alpha

    return best, best_score


# Worker wrapper for ProcessPoolExecutor (starmap-like)
def _evaluate_child_worker(args):
    """
    args: (layout, sa_prob, sa_iters, sa_t0, sa_alpha, eval_fn, seed)
    Returns (layout, score)
    """
    layout, sa_prob, sa_iters, sa_t0, sa_alpha, eval_fn, seed = args
    rng = np.random.default_rng(seed)
    if rng.random() < sa_prob:
        best_layout, best_score = simulated_annealing_local_search_worker(
            layout, eval_fn, iters=sa_iters, t0=sa_t0, alpha=sa_alpha, seed=rng.integers(
                0, 2**31 - 1)
        )
        return best_layout, best_score
    else:
        score = evaluate_layout(layout, eval_fn)
        return layout, score


# ---------------------------
# Parallel Memetic + SA main
# ---------------------------

def memetic_sa(pop_size=40, gens=200, elite_frac=0.1,
               n_characters=26, keys=30,
               crossover_rate=0.9, mutation_rate=0.2,
               sa_prob=0.5, sa_iters=300, sa_t0=1.0, sa_alpha=0.98,
               tourn_k=3, seed=42, eval_fn=None, n_workers=None,
               verbose=True):
    """
    Parallel Memetic Algorithm + SA.
    - n_workers: number of processes (None -> os.cpu_count()).
    - verbose: if True, prints initial best, per-generation improvements, and periodic progress.
    """
    rng = np.random.default_rng(seed)
    n_slots = int(keys)
    n_letters = int(n_characters)
    if n_slots < n_letters:
        raise ValueError("keys must be >= number of characters")

    if n_workers is None:
        n_workers = os.cpu_count() or 1
        print(f'Workers: {n_workers}')

    # initialize population
    pop = population_init(pop_size, n_slots, n_letters, rng)

    # initial evaluation (parallel)
    initial_args = [(pop[i], 0.0, 0, 0.0, 0.0, eval_fn,
                     rng.integers(0, 2**31 - 1)) for i in range(pop_size)]
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(_evaluate_child_worker, initial_args))
    scores = np.array([score for (_layout, score) in results])

    # global best
    best_idx = int(np.argmin(scores))
    global_best = pop[best_idx].copy()
    global_score = float(scores[best_idx])
    global BEST_LAYOUT
    BEST_LAYOUT = layout_to_letters(apply_layout(global_best, KEYS), KEYMAP)
    if verbose:
        print("Initial best:")
        print(tabulate(BEST_LAYOUT, tablefmt="fancy_grid"))
        print(f"score: {global_score:.6f}")

    n_elite = max(1, int(np.ceil(pop_size * elite_frac)))

    for gen in range(1, gens + 1):
        # keep elites
        elite_idxs = np.argsort(scores)[:n_elite]
        new_pop = [pop[idx].copy() for idx in elite_idxs]

        # produce offspring (non-evaluated yet)
        offspring = []
        offspring_seeds = []
        while len(offspring) < pop_size - n_elite:
            parent_a = tournament_select(pop, scores, rng, k=tourn_k)
            parent_b = tournament_select(pop, scores, rng, k=tourn_k)
            if rng.random() < crossover_rate:
                child = segment_crossover(parent_a, parent_b, rng)
            else:
                child = parent_a.copy()
            child = swap_mutation(child, rng, swap_prob=mutation_rate)
            offspring.append(child)
            offspring_seeds.append(int(rng.integers(0, 2**31 - 1)))

        # prepare args for parallel evaluation (apply SA per-child probabilistically in worker)
        worker_args = [
            (offspring[i], sa_prob, sa_iters, sa_t0,
             sa_alpha, eval_fn, offspring_seeds[i])
            for i in range(len(offspring))
        ]

        # evaluate offspring in parallel
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(_evaluate_child_worker, worker_args))

        # collect evaluated offspring (layout, score)
        evaluated_offspring = results  # list of (layout, score)

        # build new population (elites + evaluated offspring)
        for layout, score in evaluated_offspring:
            new_pop.append(layout)

        # trim to pop_size
        pop = np.vstack(new_pop)[:pop_size]

        # compute scores for new population (parallel)
        score_args = [(pop[i], 0.0, 0, 0.0, 0.0, eval_fn, int(
            rng.integers(0, 2**31 - 1))) for i in range(pop.shape[0])]
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(_evaluate_child_worker, score_args))
        scores = np.array([score for (_layout, score) in results], dtype=float)

        # update global best
        best_idx = int(np.argmin(scores))
        if scores[best_idx] < global_score:
            global_score = float(scores[best_idx])
            global_best = pop[best_idx].copy()
            BEST_LAYOUT = layout_to_letters(
                apply_layout(global_best, KEYS), KEYMAP)
            if verbose:
                print(f"Gen {gen}: new global best = {global_score:.6f}")
                print(tabulate(BEST_LAYOUT, tablefmt="fancy_grid"))

        # diversity reseed if collapsed
        mean_std = pop.reshape(pop_size, -1).std(axis=0).mean()
        if mean_std < 1e-3:
            num_reseed = max(1, pop_size // 2)
            for r_idx in rng.choice(pop_size, size=num_reseed, replace=False):
                pop[r_idx] = random_layout(n_slots, n_letters, rng)
            # recompute scores (parallel)
            score_args = [(pop[i], 0.0, 0, 0.0, 0.0, eval_fn, int(
                rng.integers(0, 2**31 - 1))) for i in range(pop.shape[0])]
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                results = list(ex.map(_evaluate_child_worker, score_args))
            scores = np.array(
                [score for (_layout, score) in results], dtype=float)

        if verbose and (gen % max(1, gens // 10) == 0):
            print(f"[Gen {gen}/{gens}] current global best: {global_score:.6f}")

    if verbose:
        print("Final best global score:", global_score)
    return BEST_LAYOUT, global_score
