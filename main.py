from parameters import Parameters
from GA import GA, RunLog
from problem import Problem
import typer
from individual import Individual
from load_tsp import get_tsp_data
from multiprocessing import cpu_count
import numpy as np
import deal
import ray
import csv
from pathlib import Path
from save_final_runs import save_final_runs
from graphing import graph_avg_avg_fitness, graph_max_avg_fitness, \
    graph_avg_max_fitness, graph_max_max_fitness, \
    graph_avg_avg_tour_length, graph_max_avg_tour_length, \
    graph_avg_max_tour_length, graph_max_max_tour_length


app = typer.Typer()


@ray.remote
def single_run(problem: Problem,
               parameters: Parameters,
               run_number: int = 0,
               allow_convergence: bool = True) -> RunLog:
    ga = GA(
        parameters.perc_default_min_distance,
        run_number
    )
    results = ga.run(
        problem,
        population_size=parameters.population_size,
        max_generations=parameters.max_generations,
        mutation_rate=parameters.mutation_rate,
        num_of_nodes=problem.DIMENSION,
        allow_convergence=allow_convergence)
    print(f"Run {run_number} finished.")
    results.run_number = run_number
    return results


def parameter_search(problem: Problem, 
                     runs: int = 100, 
                     save: bool = True) -> Individual:
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

    run_results: list[RunLog] = [
        single_run.options().remote(problem, parameters[i], i) for i in range(runs)  # noqa
    ]
    while run_results:
        ready, run_results = ray.wait(run_results)
        results.extend(ray.get(ready))
    best_index = 0
    final_gens: list[int] = [result.final_gen for result in results]
    results: list[Individual] = [result.best for result in results]
    for i in range(1, len(results)):
        if results[i].fitness > results[best_index].fitness:
            best_index = i
    best = results[best_index]
    best_parameters = parameters[best_index]
    print("Problem: ", problem.NAME)
    print("Best parameters: ", best_parameters)
    print("Best result: ", best.fitness)
    if save:
        print("Saving results per parameter...")
        with open(f"results/results_{problem.NAME}.csv", "w") as file:
            file.write("population_size,max_generations,mutation_rate,perc_default_min_distance,gens_to_reach_best,best_fitness\n")  # noqa
            for i in range(len(results)):
                file.write(f"{parameters[i].population_size},{parameters[i].max_generations},{parameters[i].mutation_rate},{parameters[i].perc_default_min_distance},{final_gens[i]},{results[i].fitness}\n")  # noqa
    return best


def final_runs(problem: Problem,
               parameters: Parameters,
               runs: int = 100,
               save: bool = True):
    # 100 runs with the best parameters
    # Need to get:
    #   Graphs of avg-avg fitness and max-avg fitness versus generations
    #   Graphs of avg-avg tour length and max-avg tour length versus generations # noqa
    # Which means I need to get the data from each run:
    #  avg fitness and max fitness per generation
    #  avg tour length and max tour length per generation
    #  generations to reach best fitness
    #  best fitness and best tour length per run
    #  avg fitness and avg tour length per run
    #  perc of optimal fitness and multiple of optimal tour length per run
    #  number of evaluations (fitness function calls) per run that
    #    are within some percentage of the optimal fitness
    save_path = Path(f"final_results/{problem.NAME}")
    results: list[RunLog] = []
    run_results: list[RunLog] = [
        single_run.options().remote(problem, parameters, i, allow_convergence=False) for i in range(runs)  # noqa
    ]
    while run_results:
        ready, run_results = ray.wait(run_results)
        results.extend(ray.get(ready))
    if not save:
        return
    # Save the results so I don't have to run them again
    save_final_runs(problem, parameters, results, save_path)
    # Graphs
    graph_avg_avg_fitness(problem, parameters, results, save_path)
    graph_max_avg_fitness(problem, parameters, results, save_path)
    graph_avg_max_fitness(problem, parameters, results, save_path)
    graph_max_max_fitness(problem, parameters, results, save_path)
    graph_avg_avg_tour_length(problem, parameters, results, save_path)
    graph_max_avg_tour_length(problem, parameters, results, save_path)
    graph_avg_max_tour_length(problem, parameters, results, save_path)
    graph_max_max_tour_length(problem, parameters, results, save_path)


@app.command("parameter-search")
def parameter_search_main(debug: bool = False,
                          runs: int = 100,
                          save: bool = False,
                          problem: str | None = None):
    # "data/berlin52.tsp", "data/burma14.tsp", "data/eil51.tsp",
    #            "data/eil76.tsp", "data/lin105.tsp", "data/lin318.tsp",
    #            "data/ulysses16.tsp"
    problems = ["data/berlin52.tsp", "data/burma14.tsp", "data/eil51.tsp",
                "data/eil76.tsp", "data/lin105.tsp", "data/lin318.tsp",
                "data/ulysses16.tsp"]
    # 7542, 3323, 426, 538, 14379, 42029, 6859
    optimals = [7542, 3323, 426, 538, 14379, 42029, 6859]
    if problem is not None:
        # Find index of problem
        index = -1
        for i, prob in enumerate(problems):
            if prob == problem:
                index = i
                break
        if index == -1:
            print("Invalid problem.")
            return
        problems = [problems[index]]
        optimals = [optimals[index]]

    assert len(problems) == len(optimals)
    if debug:
        deal.enable()
        ray.init(num_cpus=1)
    else:
        deal.disable()
        ray.init(num_cpus=cpu_count() - 2)
    for i, problem in enumerate(problems):
        problem: Problem = get_tsp_data(problem)
        problem.SOLUTION_COST = optimals[i]
        best = parameter_search(problem, runs, save)
        print(f"Problem: {problem.NAME}")
        print(f"Best fitness: {best.fitness}")
        print(f"Best tour length: {best.cost}")
        print(f"Optimal fitness: {problem.get_max_fitness()}")
        print(f"Optimal tour length: {problem.get_min_cost()}")
        print(f"Percentage of optimal fitness: {best.fitness / problem.get_max_fitness()}")  # noqa
        print(f"Multiple of optimal tour length: {best.cost / problem.get_min_cost()}")  # noqa
        print("")

    ray.shutdown()


@app.command("final-runs")
def final_runs_main(debug: bool = False,
                    runs: int = 100,
                    save: bool = True,
                    problem: str | None = None):
    # "data/berlin52.tsp", "data/burma14.tsp", "data/eil51.tsp",
    #            "data/eil76.tsp", "data/lin105.tsp", "data/lin318.tsp",
    #            "data/ulysses16.tsp"
    problems = ["data/berlin52.tsp", "data/burma14.tsp", "data/eil51.tsp",
                "data/eil76.tsp", "data/lin105.tsp", "data/lin318.tsp",
                "data/ulysses16.tsp"]
    # 7542, 3323, 426, 538, 14379, 42029, 6859
    optimals = [7542, 3323, 426, 538, 14379, 42029, 6859]
    if problem is not None:
        # Find index of problem
        index = -1
        for i, prob in enumerate(problems):
            if prob == problem:
                index = i
                break
        if index == -1:
            print("Invalid problem.")
            return
        problems = [problems[index]]
        optimals = [optimals[index]]

    # Load optimal parameters for each problem
    parameters: list[Parameters] = []

    with open("optimal_parameters.csv") as file:
        csvreader = csv.DictReader(file)
        for row in csvreader:
            parameters.append(Parameters(
                int(row["population_size"]),
                int(row["max_generations"]),
                float(row["mutation_rate"]),
                float(row["perc_default_min_distance"])
            ))
            if parameters[-1].max_generations < 1.5 * parameters[-1].population_size:  # noqa
                parameters[-1].max_generations = parameters[-1].population_size * 1.5  # noqa

    assert len(problems) == len(optimals)
    if debug:
        deal.enable()
        ray.init(num_cpus=1)
    else:
        deal.disable()
        ray.init(num_cpus=cpu_count() - 2)

    for i, problem in enumerate(problems):
        problem: Problem = get_tsp_data(problem)
        problem.SOLUTION_COST = optimals[i]
        final_runs(problem, parameters[i], runs, save)

@app.command("calculate-quality")
def calculate_quality_main():
    problems = ["data/berlin52.tsp", "data/burma14.tsp", "data/eil51.tsp",
                "data/eil76.tsp", "data/lin105.tsp", "data/lin318.tsp",
                "data/ulysses16.tsp"]
    optimals = [7542, 3323, 426, 538, 14379, 42029, 6859]

    for i, problem in enumerate(problems):
        problem: Problem = get_tsp_data(problem)
        problem.SOLUTION_COST = optimals[i]

        # Load the results
        # Quality is percentage distance from optimum (Average of best over all runs) # noqa
        # Reliability is percentage of runs you get within Quality -- out of total number of runs. # noqa
        # Speed is average number of evaluations needed to get within Quality. # noqa

        optimum_thresholds = [0.01, 0.02, 0.05, 0.10, 0.20]
        # Quality - distance from optimum
        quality = 0.0
        best_cost = []
        with open(f"final_results/{problem.NAME}/best_tour_length.csv") as file:
            csvreader = csv.DictReader(file)
            for row in csvreader:
                best_cost.append(float(row["best_tour_length"]))
            quality = sum(best_cost) / len(best_cost)
            quality /= problem.get_min_cost()
            quality = quality - 1.0

        # Reliability - percentage of runs within quality
        reliabilities = [0.0] * len(optimum_thresholds)
        for i, threshold in enumerate(optimum_thresholds):
            for cost in best_cost:
                if cost <= problem.SOLUTION_COST * (1.0 + threshold):
                    reliabilities[i] += 1.0
            reliabilities[i] /= len(best_cost)

        # Speed - average number of evaluations needed to get within quality
        speeds = [0.0] * len(optimum_thresholds)
        for i, threshold in enumerate(optimum_thresholds):
            for j in range(len(best_cost)):
                evals = []
                with open(f"final_results/{problem.NAME}/run_{j}/evaluations.csv") as file: # noqa
                    csvreader = csv.DictReader(file)
                    for row in csvreader:
                        evals.append(int(row["evaluations"]))
                tour_length_per_gen = []
                with open(f"final_results/{problem.NAME}/run_{j}/tour_length.csv") as file:
                    csvreader = csv.DictReader(file)
                    for row in csvreader:
                        tour_length_per_gen.append(float(row["best_tour_length"]))
                for k in range(len(evals)):
                    if tour_length_per_gen[k] <= problem.SOLUTION_COST * (1.0 + threshold):
                        speeds[i] += evals[k]
                        break
            speeds[i] /= len(best_cost)

        with open(f"final_results/{problem.NAME}/quality.csv", "w") as file:
            file.write("quality,reliability,speed\n")
            for i in range(len(optimum_thresholds)):
                file.write(f"{quality},{reliabilities[i]},{speeds[i]}\n")


@app.command("regraph")
def regraph():
    problems = ["data/berlin52.tsp", "data/burma14.tsp", "data/eil51.tsp",
                "data/eil76.tsp", "data/lin105.tsp", "data/lin318.tsp"]
    optimals = [7542, 3323, 426, 538, 14379, 42029]

    with open("optimal_parameters.csv") as file:
        csvreader = csv.DictReader(file)
        parameters: dict[str, Parameters] = {}
        for row in csvreader:
            parameters[row["problem_name"]] = Parameters(
                int(row["population_size"]),
                int(row["max_generations"]),
                float(row["mutation_rate"]),
                float(row["perc_default_min_distance"])
            )

    for i, problem in enumerate(problems):
        problem: Problem = get_tsp_data(problem)
        problem.SOLUTION_COST = optimals[i]
        save_path = Path(f"final_results/{problem.NAME}")
        results: list[RunLog] = [None] * 100

        # Load the result from each run
        for j in range(100):
            results[j] = RunLog()
            with open(f"final_results/{problem.NAME}/run_{j}/fitness.csv") as file:
                csvreader = csv.DictReader(file)
                for row in csvreader:
                    results[j].generation.append(int(row["generation"]))
                    results[j].avg_fitness.append(float(row["avg_fitness"]))
                    results[j].best_fitness.append(float(row["max_fitness"]))
            with open(f"final_results/{problem.NAME}/run_{j}/tour_length.csv") as file:
                csvreader = csv.DictReader(file)
                for row in csvreader:
                    results[j].avg_cost.append(float(row["avg_tour_length"]))
                    results[j].best_cost.append(float(row["best_tour_length"]))
            with open(f"final_results/{problem.NAME}/run_{j}/evaluations.csv") as file:
                csvreader = csv.DictReader(file)
                for row in csvreader:
                    results[j].evals.append(int(row["evaluations"]))
        # Remove all generations past the max generation
        for j in range(100):
            while results[j].generation[-1] > parameters[problem.NAME].max_generations:
                results[j].generation.pop()
                results[j].avg_fitness.pop()
                results[j].best_fitness.pop()
                results[j].avg_cost.pop()
                results[j].best_cost.pop()
                results[j].evals.pop()

        graph_avg_avg_fitness(problem, parameters[problem.NAME], results, save_path) # noqa
        graph_max_avg_fitness(problem, parameters[problem.NAME], results, save_path)
        graph_avg_max_fitness(problem, parameters[problem.NAME], results, save_path)
        graph_max_max_fitness(problem, parameters[problem.NAME], results, save_path)
        graph_avg_avg_tour_length(problem, parameters[problem.NAME], results, save_path)
        graph_max_avg_tour_length(problem, parameters[problem.NAME], results, save_path)
        graph_avg_max_tour_length(problem, parameters[problem.NAME], results, save_path)
        graph_max_max_tour_length(problem, parameters[problem.NAME], results, save_path)

if __name__ == "__main__":
    app()
