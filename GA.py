from individual import Individual
from problem import Problem
from coding import decode
import numpy as np
from enum import Enum

generator = np.random.default_rng()


class IncestPreventionMethod(Enum):
    HAMMING_DISTANCE = "hamming_distance"
    EDGEWISE_DISTANCE = "edgewise_distance"


class GA():
    """
    The genetic algorithm.

    Args:
        incest_prevention_method (IncestPreventionMethod): The method of
            incest prevention.
        perc_default_min_distance (float): The percentage of the maximum
            possible distance between two individuals in the population
            to use as the default minimum distance for crossover.
            If None, the default minimum distance is
            len(population[0].get_chromosome()) / 4.
    """
    def __init__(self,
                 incest_prevention_method=IncestPreventionMethod.EDGEWISE_DISTANCE,  # noqa
                 perc_default_min_distance: float | None = None,
                 id: int = 0):
        self._incest_prevention_method = incest_prevention_method
        assert 0 <= perc_default_min_distance <= 1 or \
            perc_default_min_distance is None
        self.perc_default_min_distance = perc_default_min_distance
        self.id = id

    def initial_distance(self, population: list[Individual]) -> float:
        """
        Calculate the initial distance between the individuals in the
        population.

        Args:
            population (list[Individual]): The population.

        Returns:
            float: The initial distance.
        """
        if self.perc_default_min_distance is not None:
            # Return the percentage of the maximum possible distance
            # Max possible distance is the number of edges in the
            # chromosome (number of genes - 1).
            return self.perc_default_min_distance * \
                (population[0].get_number_of_genes() - 1)
        return len(population[0].get_chromosome()) / 4

    def incest_prevention(
            self,
            parent1: Individual, parent2: Individual,
            min_distance: float) -> bool:
        """
        Prevent inbreeding.

        Args:
            parent1 (Individual): The first parent.
            parent2 (Individual): The second parent.
            min_distance (float): The minimum distance between the parents.

        Returns:
            bool: True if the parents are too similar, False otherwise.
        """
        if self._incest_prevention_method == IncestPreventionMethod.HAMMING_DISTANCE: # noqa
            return parent1.get_chromosome().hamming_distance(
                parent2.get_chromosome()) / 2 < min_distance
        elif self._incest_prevention_method == IncestPreventionMethod.EDGEWISE_DISTANCE: # noqa
            return parent1.get_chromosome().edgewise_distance(
                parent2.get_chromosome()) < min_distance

    def evaluate(self,
                 population: list[Individual],
                 problem: Problem):
        """
        Evaluate the fitness of the population.

        Args:
            population (list[Individual]): The population to evaluate.
            problem (Problem): The problem to solve.
        """
        for individual in population:
            tour = decode(individual.get_chromosome())
            fitness, cost = problem.evaluate(tour)
            individual.set_fitness(fitness)
            individual.set_cost(cost)

    def crossover(self,
                  parents: list[Individual],
                  min_distance: int = 1) -> list[Individual]:
        """
        Perform crossover on the parents.

        Args:
            parents (list[Individual]): The parents to crossover.
            min_distance (int): The minimum distance for
                crossover. Prevents inbreeding.

        Returns:
            list[Individual]: The children.
        """
        children = []
        # Shuffle parents
        generator.shuffle(parents)
        for i in range(0, len(parents), 2):
            # Check if parents are too similar
            if self.incest_prevention(
                parents[i],
                parents[i + 1],
                min_distance
            ):
                continue
            children.append(parents[i].crossover(parents[i + 1]))
        return children

    def mutate(self,
               population: list[Individual],
               mutation_rate: float = 0.01) -> list[Individual]:
        """
        Mutate the population.

        Args:
            population (list[Individual]): The population to mutate.
            mutation_rate (float): The probability of mutation.

        Returns:
            list[Individual]: The mutated population.
        """
        for individual in population:
            individual.mutate(mutation_rate)
        return population

    def run(self,
            problem: Problem,
            population_size: int = 100,
            max_generations: int = 100,
            mutation_rate: float = 0.01,
            num_of_nodes: int = 10) -> tuple[Individual, int]:
        """
        Args:
            problem (Problem): The problem to solve.
            population_size (int): The size of the population.
            max_generations (int): The maximum number of generations.
            mutation_rate (float): The probability of mutation.
            num_of_nodes (int): The number of nodes in the TSP.

        Returns:
            Individual: The best individual.
            int: The number of generations.
        """
        # Initialize population with random individuals
        # and evaluate fitness
        population = [Individual(num_of_nodes) for _ in range(population_size)]
        self.evaluate(population, problem)

        generation = 0
        prev_gen_stats = (-1, -1, -1)
        gens_without_change = 0
        min_distance = self.initial_distance(population)
        # Main loop
        while generation < max_generations:
            # Crossover
            children = self.crossover(population,
                                      min_distance)
            # Check if crossover produced children
            # If not, decrease min_distance
            if len(children) == 0:
                min_distance -= 1
            # Evaluate fitness
            self.evaluate(children, problem)
            # Select the best individuals from the population
            # and the children
            population = sorted(population + children,
                                key=lambda x: x.get_fitness(),
                                reverse=True)[:population_size]
            # Mutation
            if min_distance <= 0:
                # Reset min_distance, keep
                # the best individual, and mutate the
                # rest of the population
                min_distance = self.initial_distance(population)
                population = sorted(population,
                                    key=lambda x: x.get_fitness(),
                                    reverse=True)
                population[1:] = self.mutate(population[1:], mutation_rate)
            elif gens_without_change > 0.1 * max_generations and gens_without_change > 100: # noqa
                print("Converged!")
                break

            # Check if the population changed
            best_fitness = population[0].get_fitness()
            avg_fitness = np.mean([i.get_fitness() for i in population])
            worst_fitness = population[-1].get_fitness()
            if (best_fitness, avg_fitness, worst_fitness) == prev_gen_stats:
                gens_without_change += 1
            else:
                gens_without_change = 0

            def get_unique_chromosomes(population: list[Individual]):
                res = []
                for individual in population:
                    chromosome = str(individual.get_chromosome())
                    if chromosome not in res:
                        res.append(chromosome)
                return res

            prev_gen_stats = (best_fitness, avg_fitness, worst_fitness)
            generation += 1
            if generation % 100 == 0:
                print(f"Run {self.id} generation {generation}: {best_fitness}")
        # end while
        # Return the best individual and the number of generations
        return (sorted(population,
                       key=lambda x: x.get_fitness(),
                       reverse=True)[0],
                generation)
