from os import cpu_count
from tabulate import tabulate
from concurrent.futures import ProcessPoolExecutor
from optimizer.evaluate import evaluate
from utils import apply_layout, layout_to_letters
from config import LAYOUT_TEMPLATE, KEYMAP
import numpy as np


def evaluate_layout(layout):
    return evaluate(layout)


def random_layout(n_slots, n_letters, rng):
    """Create a random layout array (slot->letter values 0..n_letters)."""
    slots = np.arange(n_slots)
    rng.shuffle(slots)
    layout = np.zeros(n_slots, dtype=int)
    for letter_idx, slot in enumerate(slots[:n_letters], start=1):
        layout[slot] = letter_idx
    return layout


def init_population(n_pop, n_slots, n_letters, rng):
    """Initialize population of slot->letter layouts."""
    pop = np.zeros((n_pop, n_slots), dtype=int)
    for i in range(n_pop):
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


def inversion_mutation(layout, rng, max_span=10):
    new = layout.copy()
    n = len(new)
    i, j = sorted(rng.integers(0, n, size=2))
    new[i:j+1] = new[i:j+1][::-1]
    return new


def swap_mutation(layout, rng, swap_prob=0.1):
    if rng.random() < 0.5:
        return inversion_mutation(layout, rng)
    new = layout.copy()
    n = len(new)
    attempts = max(1, int(np.ceil(n * swap_prob)))
    for _ in range(attempts):
        i, j = rng.integers(0, n, size=2)
        new[i], new[j] = new[j], new[i]
    return new


def ensure_unique_mutation(layout, seen_set, rng,
                           swap_prob=0.5, max_attempts=100):
    for _ in range(max_attempts):
        candidate = swap_mutation(layout, rng, swap_prob=swap_prob)

        tup = tuple(int(x) for x in candidate.tolist())
        if tup not in seen_set:
            seen_set.add(tup)
            return candidate

    n_slots = len(layout)
    n_char = int(np.count_nonzero(layout))

    while True:
        candidate = random_layout(n_slots, n_char)
        tup = tuple(int(x) for x in candidate.tolist())
        if tup not in seen_set:
            seen_set.add(tup)
            return candidate


def simulated_annealing(layout: list,   n_iter: int,
                        f_t0: float,    f_alpha: float,
                        seed=None):
    rng = np.random.default_rng(seed)
    current_layout = layout.copy()
    current_metrics, current_score = evaluate_layout(current_layout)
    best = current_layout.copy()
    best_metrics = current_metrics
    best_score = current_score
    T = f_t0
    n_slots = len(current_layout)

    for _ in range(n_iter):
        i, j = rng.integers(0, n_slots, size=2)
        neighbor = current_layout.copy()
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        neigh_metrics, neigh_score = evaluate_layout(neighbor)
        delta = ((neigh_score/best_score)*100) - ((current_score/best_score)*100)
        if delta < 0 or rng.random() < np.exp(-delta / max(1e-12, T)):
            current_metrics, current_score, current = (neigh_metrics,
                                                       neigh_score,
                                                       neighbor)
            if current_score < best_score:
                best_metrics = current_metrics
                best_score = current_score
                best = current
        T *= f_alpha

    return best_metrics, best_score, best


def _evaluate_child_worker(args):
    layout, f_prob, n_iter, f_t0, f_alpha, seed = args
    rng = np.random.default_rng(seed)
    if rng.random() < f_prob:
        best_metrics, best_score, best_layout = simulated_annealing(
            layout, n_iter, f_t0, f_alpha, seed)
        return best_metrics, best_score, best_layout
    else:
        metrics, score = evaluate_layout(layout)
        return metrics, score, layout


def msa(
        n_pop: int,     n_gen:  int,
        n_char: int,    n_slots: int,
        f_elite: float = 0.05,  f_mut: float = 0.15,
        f_cross: float = 0.9,   f_prob: float = 0.5,
        n_iter: int = 500,      f_t0: float = 1.0,
        f_alpha: float = 0.98,  n_tourn: int = 3,
        seed: int = 42,         n_workers: int = None,
        verbose: bool = True
) -> [[], int]:
    """
    Memetic + Simulated Annealing
    """
    rng = np.random.default_rng(seed)
    if n_char > n_slots:
        raise ValueError("Available slots must be >= number of characters")

    n_workers = cpu_count() or 1 if n_workers is None else n_workers

    pop = init_population(n_pop, n_slots, n_char, rng)

    seen_layouts = set(tuple(int(x) for x in row.tolist()) for row in pop)

    initial_args = [(pop[i], 0.0, 0, 0.0, 0.0,
                     rng.integers(0, 2**31 - 1)) for i in range(n_pop)]
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(_evaluate_child_worker, initial_args))

    scores = np.array([score for (_metrics, score, _layout) in results])
    print(scores)
    # l_metrics = np.array([metrics for (metrics, _score, _layout) in results])

    best_idx = int(np.argmin(scores))
    global_best = pop[best_idx].copy()
    # global_metrics = l_metrics[best_idx].copy()
    global_score = float(scores[best_idx])

    if verbose:
        print(f"Initial best score: {global_score:.6f}")

    # number of elites
    n_elite = max(1, int(np.ceil(n_pop * f_elite)))

    # main loop
    for gen in range(1, n_gen + 1):
        # keep elites (best n_elite individuals)
        elite_idxs = np.argsort(scores)[:n_elite]
        new_pop = [pop[idx].copy() for idx in elite_idxs]

        # produce offspring until we fill remaining slots
        offspring = []
        offspring_seeds = []
        while len(offspring) < (n_pop - n_elite):
            parent_a = tournament_select(pop, scores, rng, k=n_tourn)
            parent_b = tournament_select(pop, scores, rng, k=n_tourn)
            if rng.random() < f_cross:
                child = segment_crossover(parent_a, parent_b, rng)
            else:
                child = parent_a.copy()

            child = ensure_unique_mutation(
                child, seen_layouts, rng, swap_prob=f_mut)

            offspring.append(child)
            offspring_seeds.append(int(rng.integers(0, 2**31 - 1)))

        worker_args = [
            (offspring[i], f_prob, n_iter, f_t0, f_alpha, offspring_seeds[i])
            for i in range(len(offspring))
        ]

        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(_evaluate_child_worker, worker_args))

        for _metrics, _score, layout in results:
            new_pop.append(layout)

        pop = np.vstack(new_pop)[:n_pop]

        seen_layouts = set(tuple(int(x) for x in row.tolist()) for row in pop)

        score_args = [
            (pop[i], 0.0, 0, 0.0, 0.0, int(rng.integers(0, 2**31 - 1)))
            for i in range(pop.shape[0])
        ]
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(_evaluate_child_worker, score_args))
        scores = np.array(
            [score for (_metrics, score, _layout) in results], dtype=float)
        # l_metrics = np.array([metrics for (metrics, _score, _layout) in results], dtype=float)

        # update global best
        best_idx = int(np.argmin(scores))
        if scores[best_idx] < global_score:
            global_score = float(scores[best_idx])
            global_best = pop[best_idx].copy()
            best_layout = layout_to_letters(
                apply_layout(global_best, LAYOUT_TEMPLATE), KEYMAP)
            print(tabulate(best_layout, tablefmt="fancy_grid"))
            # global_metrics = l_metrics[best_idx].copy()
            if verbose:
                print(f"Gen {gen}: new global best = {global_score:.6f}")

        mean_std = pop.reshape(n_pop, -1).std(axis=0).mean()
        if mean_std < 1e-3:
            num_reseed = max(1, n_pop // 2)
            for r_idx in rng.choice(n_pop, size=num_reseed, replace=False):
                pop[r_idx] = random_layout(n_slots, n_char, rng)
                seen_layouts.add(tuple(int(x) for x in pop[r_idx].tolist()))
            score_args = [
                (pop[i], 0.0, 0, 0.0, 0.0, int(rng.integers(0, 2**31 - 1)))
                for i in range(pop.shape[0])
            ]
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                results = list(ex.map(_evaluate_child_worker, score_args))
            scores = np.array(
                [score for (_metrics, score, _layout) in results], dtype=float)

        if verbose and (gen % max(1, n_gen // 10) == 0):
            print(
                f"[Gen {gen}/{n_gen}] current global best: {global_score:.6f}")

    if verbose:
        print("Final best global score:", global_score)

    return global_best, global_score
