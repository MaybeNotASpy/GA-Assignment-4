from dataclasses import dataclass
from GA import GA, IncestPreventionMethod
from problem import Problem
from coding import decode
from individual import Individual
from load_tsp import get_tsp_data, get_tour_data
from multiprocessing import Pool, cpu_count
import numpy as np
import deal


@dataclass
class Parameters():
    population_size: int
    max_generations: int
    mutation_rate: float
    perc_default_min_distance: float


def single_run(problem: Problem,
               parameters: Parameters,
               run_number: int = 0) -> Individual:
    ga = GA(
        IncestPreventionMethod.EDGEWISE_DISTANCE,
        parameters.perc_default_min_distance,
        run_number
    )
    best, final_gen = ga.run(
        problem,
        population_size=parameters.population_size,
        max_generations=parameters.max_generations,
        mutation_rate=parameters.mutation_rate,
        num_of_nodes=problem.DIMENSION)
    print(f"Run {run_number} finished.")
    return (best, final_gen)


def parameter_search(problem: Problem, runs: int = 100) -> Individual:
    parameters: list[Parameters] = [None] * runs
    generator = np.random.default_rng()
    population_size_values = generator.integers(5, 500, size=runs)
    population_size_values *= 2
    max_generations_values = generator.integers(100, 1000, size=runs)
    mutation_rate_values = generator.uniform(0.01, 1., size=runs)
    perc_default_min_distance_values = generator.uniform(0.01, 1., size=runs)
    for i in range(runs):
        parameters[i] = Parameters(
            population_size_values[i],
            max_generations_values[i],
            mutation_rate_values[i],
            perc_default_min_distance_values[i]
        )
    best = None
    best_parameters = None
    results = []
    with Pool(processes=cpu_count()//2) as pool:
        for i in range(runs):
            results.append(
                pool.apply_async(single_run, (problem, parameters[i], i))
            )
        results = [result.get() for result in results]
    best_index = 0
    for i in range(1, len(results)):
        if results[i][0].get_fitness() > results[best_index][0].get_fitness():
            best_index = i
    best, final_gen = results[best_index]
    results = [result[0] for result in results]
    best_parameters = parameters[best_index]
    print("Problem: ", problem.NAME)
    print("Best parameters: ", best_parameters)
    print("Best result: ", best.get_fitness())
    print("Saving results per parameter...")
    with open(f"results/results_{problem.NAME}.csv", "w") as file:
        file.write("population_size,max_generations,mutation_rate,perc_default_min_distance,gens_to_reach_best,best_fitness\n")  # noqa
        for i in range(len(results)):
            file.write(f"{parameters[i].population_size},{parameters[i].max_generations},{parameters[i].mutation_rate},{parameters[i].perc_default_min_distance},{final_gen},{results[i].get_fitness()}\n")  # noqa
    return best


if __name__ == "__main__":
    # "data/berlin52.tsp", "data/burma14.tsp",
    problems = ["data/eil51.tsp",
                "data/eil76.tsp", "data/lin105.tsp", "data/lin318.tsp",
                "data/ulysses16.tsp"]
    # 7542, 3323, 
    optimals = [426, 538, 14379, 42029, 6859]
    deal.disable()
    for i, problem in enumerate(problems):
        problem = get_tsp_data(problem)
        problem.SOLUTION_COST = optimals[i]
        best = parameter_search(problem)
        print(f"Problem: {problem.NAME}")
        print(best)
        print(f"Best cost: {best.get_cost()}")
        print(f"Best fitness: {best.get_fitness()}")
        final_result = decode(best.get_chromosome())
        print(f"Final result: {final_result}")
        print(f"Max fitness: {problem.get_max_fitness()}")
        print(f"Min cost: {problem.get_min_cost()}")
        print(f"Percentage of optimal fitness: {best.get_fitness() / problem.get_max_fitness() * 100}%")  # noqa
        print(f"Multiple of optimal cost: {best.get_cost() / problem.get_min_cost()}x")  # noqa
