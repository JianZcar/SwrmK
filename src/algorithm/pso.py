import numpy as np
from algorithm.evaluate import evaluate
from utils import apply_layout
from config import KEYS, FINGER, EFFORT, DISTANCE

# ===========================
# Helper functions
# ===========================


def decode_continuous_to_layout(pos_scores):
    """Convert continuous (n_letters x n_slots) matrix into discrete layout."""
    pos = pos_scores.copy()
    n_letters, n_slots = pos.shape
    layout = np.zeros(n_slots, dtype=int)

    for _ in range(n_letters):
        flat_idx = np.argmax(pos)
        letter_idx, slot_idx = divmod(flat_idx, n_slots)
        layout[slot_idx] = 1 + letter_idx  # start_key always 1
        pos[letter_idx, :] = -np.inf
        pos[:, slot_idx] = -np.inf

    return layout


def neighbors(idx, n_particles, neighbor_k=1):
    """Return indices of ring-local neighbors for a particle."""
    out = []
    for k in range(1, neighbor_k + 1):
        out.append((idx - k) % n_particles)
        out.append((idx + k) % n_particles)
    return out


def update_local_best(i, primals, primes_cont, local_bests_cont,
                      local_best_scores, neighbor_k=1):
    """Update local best for particle i based on neighbors."""
    best_idx = i
    best_score = primals[i]
    n_particles = len(primals)
    for nb in neighbors(i, n_particles, neighbor_k):
        if primals[nb] < best_score:
            best_score = primals[nb]
            best_idx = nb
    local_bests_cont[i] = primes_cont[best_idx].copy()
    local_best_scores[i] = best_score


def initialize_particles(n_particles, n_letters, n_slots,
                         rng, max_velocity=10.0):
    """Initialize particle positions and velocities."""
    positions = rng.random((n_particles, n_letters, n_slots))
    velocities = rng.uniform(-max_velocity, max_velocity,
                             (n_particles, n_letters, n_slots))
    return positions, velocities


def evaluate_particle(position, eval_fn):
    """Decode continuous position to layout and evaluate score."""
    layout = decode_continuous_to_layout(position)
    print(layout)
    print(apply_layout(layout, KEYS))
    score = float(eval_fn(layout)) if eval_fn is not None else np.inf
    return layout, score


def update_velocity_position(positions, velocities, primes_cont,
                             local_bests_cont, w, c_cog_t, c_soc_t,
                             r_cog, r_soc, max_velocity, rng,
                             perturb_prob=0.08, perturb_scale=1.0):
    """Update velocities and positions, apply optional perturbation."""
    velocities = w * velocities + (c_cog_t * r_cog) * (primes_cont - positions) + \
        (c_soc_t * r_soc) * (local_bests_cont - positions)

    mask = rng.random(positions.shape[0]) < perturb_prob
    for i in range(positions.shape[0]):
        if mask[i]:
            velocities[i] += rng.normal(scale=perturb_scale,
                                        size=velocities[i].shape)

    velocities = np.clip(velocities, -max_velocity, max_velocity)
    positions += velocities
    positions = np.clip(positions, -10.0, 10.0)
    return positions, velocities


def evaluate_update_particle(i, positions, primals, primes_cont, primes_decoded,
                             no_improve_iters, local_bests_cont, local_best_scores,
                             global_best_cont, global_best_decoded, global_score,
                             eval_fn, neighbor_k=1):
    """Evaluate particle, update personal, local, and global bests."""
    decoded, score = evaluate_particle(positions[i], eval_fn)

    if score < primals[i]:
        primals[i] = score
        primes_cont[i] = positions[i].copy()
        primes_decoded[i] = decoded.copy()
        no_improve_iters[i] = 0
    else:
        no_improve_iters[i] += 1

    if score < global_score:
        global_score = score
        global_best_cont = positions[i].copy()
        global_best_decoded = decoded.copy()
        for nb in neighbors(i, positions.shape[0], neighbor_k):
            update_local_best(nb, primals, primes_cont,
                              local_bests_cont, local_best_scores, neighbor_k)
        update_local_best(i, primals, primes_cont,
                          local_bests_cont, local_best_scores, neighbor_k)

    update_local_best(i, primals, primes_cont,
                      local_bests_cont, local_best_scores, neighbor_k)
    return decoded, score, global_best_cont, global_best_decoded, global_score


def stagnation_reseed(i, positions, velocities, primes_cont, primes_decoded, primals,
                      no_improve_iters, rng, max_velocity,
                      local_bests_cont, local_best_scores, neighbor_k=1):
    """Re-seed a stagnated particle."""
    positions[i] = rng.random(positions[i].shape)
    velocities[i] = rng.uniform(-max_velocity,
                                max_velocity, velocities[i].shape)
    primes_cont[i] = positions[i].copy()
    primes_decoded[i] = decode_continuous_to_layout(positions[i])
    primals[i] = np.inf
    no_improve_iters[i] = 0

    for nb in neighbors(i, positions.shape[0], neighbor_k):
        update_local_best(nb, primals, primes_cont,
                          local_bests_cont, local_best_scores, neighbor_k)
    update_local_best(i, primals, primes_cont,
                      local_bests_cont, local_best_scores, neighbor_k)


def diversity_reseed(positions, velocities, primals, primes_cont, primes_decoded,
                     no_improve_iters, local_bests_cont, local_best_scores,
                     rng, reseed_fraction, max_velocity, neighbor_k=1):
    """Re-seed the worst particles if diversity is low."""
    n_particles = positions.shape[0]
    worst_count = max(1, int(np.ceil(n_particles * reseed_fraction)))
    worst_idx = np.argsort(primals)[-worst_count:]

    for idx in worst_idx:
        positions[idx] = rng.random(positions[idx].shape)
        velocities[idx] = rng.uniform(-max_velocity,
                                      max_velocity, velocities[idx].shape)
        primes_cont[idx] = positions[idx].copy()
        primes_decoded[idx] = decode_continuous_to_layout(positions[idx])
        primals[idx] = np.inf
        no_improve_iters[idx] = 0
        update_local_best(idx, primals, primes_cont,
                          local_bests_cont, local_best_scores, neighbor_k)
        for nb in neighbors(idx, n_particles, neighbor_k):
            update_local_best(nb, primals, primes_cont,
                              local_bests_cont, local_best_scores, neighbor_k)

# ===========================
# Main PSO function
# ===========================


def pso(n_particles, n_iters, c0, c1,
        n_characters=26, keys=30,
        max_velocity=10.0, seed=42, eval_fn=None):
    """PSO for keyboard-layout optimization using modular helpers."""
    rng = np.random.default_rng(seed)
    n_slots = int(keys)
    n_letters = int(n_characters)
    if n_slots < n_letters:
        raise ValueError("keys must be >= number of characters")

    # Hyperparameters
    neighbor_k = 1
    stagnation_limit = 6
    perturb_prob = 0.08
    perturb_scale = 0.1 * max_velocity
    diversity_check_every = 4
    diversity_threshold = 1e-2
    reseed_fraction = 0.4
    c_cog_init = max(c0, 1.5)
    c_cog_final = 0.5
    c_soc_init = 0.5
    c_soc_final = max(c1, 1.5)

    # Initialize particles
    positions, velocities = initialize_particles(
        n_particles, n_letters, n_slots, rng, max_velocity)
    primes_cont = positions.copy()
    primes_decoded = np.zeros((n_particles, n_slots), dtype=int)
    primals = np.full(n_particles, np.inf)
    local_bests_cont = positions.copy()
    local_best_scores = np.full(n_particles, np.inf)
    no_improve_iters = np.zeros(n_particles, dtype=int)
    global_best_cont = positions[0].copy()
    global_best_decoded = np.zeros(n_slots, dtype=int)
    global_score = np.inf

    # Initial evaluation
    for i in range(n_particles):
        decoded, score = evaluate_particle(positions[i], eval_fn)
        primals[i] = score
        primes_decoded[i] = decoded.copy()
        update_local_best(i, primals, primes_cont,
                          local_bests_cont, local_best_scores, neighbor_k)
        if score < global_score:
            global_score = score
            global_best_cont = positions[i].copy()
            global_best_decoded = decoded.copy()

    # Main loop
    w_max, w_min = 0.9, 0.4
    for t in range(n_iters):
        w = w_max - (w_max - w_min) * (t / max(1, n_iters-1))
        c_cog_t = c_cog_init + (c_cog_final - c_cog_init) * \
            (t / max(1, n_iters-1))
        c_soc_t = c_soc_init + (c_soc_final - c_soc_init) * \
            (t / max(1, n_iters-1))
        r_cog = rng.random((n_particles, n_letters, n_slots))
        r_soc = rng.random((n_particles, n_letters, n_slots))

        positions, velocities = update_velocity_position(
            positions, velocities, primes_cont, local_bests_cont,
            w, c_cog_t, c_soc_t, r_cog, r_soc, max_velocity, rng,
            perturb_prob, perturb_scale
        )

        for i in range(n_particles):
            decoded, score, global_best_cont,
            global_best_decoded, global_score = evaluate_update_particle(
                i, positions, primals, primes_cont,
                primes_decoded, no_improve_iters,
                local_bests_cont, local_best_scores,
                global_best_cont, global_best_decoded,
                global_score, eval_fn, neighbor_k
            )

            if no_improve_iters[i] >= stagnation_limit:
                stagnation_reseed(i, positions, velocities,
                                  primes_cont, primes_decoded,
                                  primals, no_improve_iters, rng, max_velocity,
                                  local_bests_cont, local_best_scores,
                                  neighbor_k)

        # Diversity check
        if t % diversity_check_every == 0:
            mean_std = positions.reshape(n_particles, -1).std(axis=0).mean()
            if mean_std < diversity_threshold:
                diversity_reseed(positions, velocities, primals,
                                 primes_cont, primes_decoded,
                                 no_improve_iters, local_bests_cont,
                                 local_best_scores, rng, reseed_fraction,
                                 max_velocity, neighbor_k)

    print("Best global score:", global_score)
    return global_best_decoded, global_score
