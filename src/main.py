# from algorithm.pso import pso
from tabulate import tabulate
from algorithm.memetic_sa import memetic_sa
from algorithm.evaluate import evaluate

# pso(100, 10, 0.3, 0.9, 26, 28, 10, 69, evaluate)


best_layout, best_score = memetic_sa(
    pop_size=100,       # number of layouts in population
    gens=50,            # number of generations
    elite_frac=0.1,     # fraction of elite layouts kept
    n_characters=26,    # number of letters to place
    keys=28,            # number of slots available
    crossover_rate=0.9,  # probability of crossover
    mutation_rate=0.2,  # swap mutation probability
    sa_prob=0.6,        # probability of applying SA to offspring
    sa_iters=50,       # SA iterations
    sa_t0=1.0,          # initial SA temperature
    sa_alpha=0.98,      # SA cooling factor
    tourn_k=3,          # tournament size
    seed=69,            # random seed
    eval_fn=evaluate    # your scoring function
)

print("Best layout found:")

print(tabulate(best_layout, tablefmt="fancy_grid"))
print("Score:", best_score)
