from dataclasses import dataclass


@dataclass
class Parameters():
    population_size: int
    max_generations: int
    mutation_rate: float
    perc_default_min_distance: float
