from individual import Individual
from problem import Problem
from coding import decode
import numpy as np
import deal
from copy import copy

generator = np.random.default_rng()


class RunLog():
    """
    A class to log the results of a run.
    """
    def __init__(self):
        self.generation: list[int] = []
        self.avg_fitness: list[float] = []
        self.best_fitness: list[float] = []
        self.avg_cost: list[float] = []
        self.best_cost: list[float] = []
        self.evals: list[int] = []
        self.best: Individual = None
        self.final_gen: int = None
        self._run_number: int = None

    def log_run(self,
                generation: int,
                avg_fitness: float,
                best_fitness: float,
                avg_cost: float,
                best_cost: float,
                evals: int):
        """
        Log the results of a generation.

        Args:
            generation (int): The generation number.
            avg_fitness (float): The average fitness.
            best_fitness (float): The best fitness.
            avg_cost (float): The average cost.
            best_cost (float): The best cost.
            evals (int): The number of evaluations.
        """
        self.generation.append(generation)
        self.avg_fitness.append(avg_fitness)
        self.best_fitness.append(best_fitness)
        self.avg_cost.append(avg_cost)
        self.best_cost.append(best_cost)
        self.evals.append(evals)

    def log_run_end(self,
                    best: Individual,
                    final_gen: int):
        """
        Log the results of the run.

        Args:
            best (Individual): The best individual.
            final_gen (int): The number of generations.
        """
        self.best = best
        self.final_gen = final_gen

    @property
    def run_number(self):
        """
        Get the run number.

        Returns:
            int: The run number.
        """
        return self._run_number

    @run_number.setter
    def run_number(self, value):
        """
        Set the run number.

        Args:
            value (int): The run number.
        """
        self._run_number = value


class GA():
    """
    The genetic algorithm.

    Args:
        perc_default_min_distance (float): The percentage of the maximum
            possible distance between two individuals in the population
            to use as the default minimum distance for crossover.
        id (int): The id of the GA.
    """
    @deal.pre(lambda self, perc_default_min_distance, id = 0: 0 <= perc_default_min_distance <= 1) # noqa
    def __init__(self,
                 perc_default_min_distance: float,
                 id: int = 0):
        self.perc_default_min_distance = perc_default_min_distance
        self.id = id

    def initial_distance(self, population: list[Individual]) -> float:
        """
        Calculate the initial distance between two individuals
        in the population.

        Args:
            population (list[Individual]): The population.

        Returns:
            float: The initial distance.
        """
        return self.perc_default_min_distance * population[0].num_of_genes

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
        return parent1.chromosome.edgewise_distance(
            parent2.chromosome) < min_distance

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
            tour = decode(individual.chromosome)
            fitness, cost = problem.evaluate(tour)
            individual.fitness = fitness
            individual.cost = cost

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
            num_of_nodes: int = 10,
            allow_convergence: bool = True) -> RunLog:
        """
        Args:
            problem (Problem): The problem to solve.
            population_size (int): The size of the population.
            max_generations (int): The maximum number of generations.
            mutation_rate (float): The probability of mutation.
            num_of_nodes (int): The number of nodes in the TSP.
            allow_convergence (bool): Whether or not to allow the
                algorithm to converge.

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
        run_log = RunLog()
        running_evals = 0
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
            running_evals += len(children)
            # Select the best individuals from the population
            # and the children
            population = sorted(population + children,
                                key=lambda x: x.fitness,
                                reverse=True)[:population_size]
            # Mutation
            if min_distance <= 0:
                # Reset min_distance, keep
                # the best individual, and mutate the
                # rest of the population
                min_distance = self.initial_distance(population)
                population = sorted(population,
                                    key=lambda x: x.fitness,
                                    reverse=True)
                population[1:] = self.mutate(population[1:], mutation_rate)
            elif allow_convergence \
                and gens_without_change > 0.1 * max_generations \
                    and gens_without_change > 100:
                print("Converged!")
                break

            best_fitness = population[0].fitness
            avg_fitness = np.mean([i.fitness for i in population])
            if allow_convergence:
                # Check if the population changed
                worst_fitness = population[-1].fitness
                if (best_fitness, avg_fitness, worst_fitness) == prev_gen_stats:
                    gens_without_change += 1
                else:
                    gens_without_change = 0

                prev_gen_stats = (best_fitness, avg_fitness, worst_fitness)
            generation += 1
            if generation % 100 == 0:
                print(f"Run {self.id} generation {generation}: {best_fitness}")

            # Log the results of the generation
            run_log.log_run(generation,
                            avg_fitness,
                            best_fitness,
                            np.mean([i.cost for i in population]),
                            population[0].cost,
                            copy(running_evals))
        # end while
        # Log the results of the run
        best = sorted(population,
                      key=lambda x: x.fitness,
                      reverse=True)[0]
        run_log.log_run_end(best, generation)
        return run_log
