from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from GA import RunLog
from problem import Problem
from parameters import Parameters


def graph_avg_avg_fitness(problem: Problem,
                          parameters: Parameters,
                          results: list[RunLog],
                          save_path: Path):
    # Calculate the average average fitness
    avg_avg_fitness = np.zeros(len(results[0].avg_fitness))
    for result in results:
        avg_fitness = np.array(result.avg_fitness)
        avg_avg_fitness += avg_fitness
    avg_avg_fitness /= len(results)
    # Plot the average average fitness
    plt.plot(results[0].generation, avg_avg_fitness)
    # Plot the optimal fitness
    plt.plot(results[0].generation,
             np.full(len(results[0].generation),
             problem.get_max_fitness()),
             linestyle="--")
    plt.legend(["Average-Average Fitness", "Optimal Fitness"])
    plt.xlabel(f"Generation (Population Size: {parameters.population_size})")
    plt.ylabel("Average-Average Fitness")
    plt.title(f"Average-Average Fitness for {problem.NAME}")
    plt.savefig(save_path / "avg_avg_fitness.pdf", dpi=300, bbox_inches = "tight")
    plt.close()


def graph_max_avg_fitness(problem: Problem,
                          parameters: Parameters,
                          results: list[RunLog],
                          save_path: Path):
    # Calculate the max average fitness
    max_avg_fitness = np.zeros(len(results[0].avg_fitness))
    for result in results:
        for i in range(len(result.avg_fitness)):
            if result.avg_fitness[i] > max_avg_fitness[i]:
                max_avg_fitness[i] = result.avg_fitness[i]
    # Plot the max average fitness
    plt.plot(results[0].generation, max_avg_fitness)
    # Plot the optimal fitness
    plt.plot(results[0].generation,
             np.full(len(results[0].generation),
             problem.get_max_fitness()),
             linestyle="--")
    plt.legend(["Max-Average Fitness", "Optimal Fitness"])
    plt.xlabel(f"Generation (Population Size: {parameters.population_size})")
    plt.ylabel("Max-Average Fitness")
    plt.title(f"Max-Average Fitness for {problem.NAME}")
    plt.savefig(save_path / "max_avg_fitness.pdf", dpi=300, bbox_inches = "tight")
    plt.close()


def graph_avg_max_fitness(problem: Problem,
                          parameters: Parameters,
                          results: list[RunLog],
                          save_path: Path):
    # Calculate the average max fitness
    avg_max_fitness = np.zeros(len(results[0].avg_fitness))
    for result in results:
        for i in range(len(result.best_fitness)):
            avg_max_fitness[i] += result.best_fitness[i]
    avg_max_fitness /= len(results)
    # Plot the average max fitness
    plt.plot(results[0].generation, avg_max_fitness)
    # Plot the optimal fitness
    plt.plot(results[0].generation,
             np.full(len(results[0].generation),
             problem.get_max_fitness()),
             linestyle="--")
    plt.legend(["Average-Max Fitness", "Optimal Fitness"])
    plt.xlabel(f"Generation (Population Size: {parameters.population_size})")
    plt.ylabel("Average-Max Fitness")
    plt.title(f"Average-Max Fitness for {problem.NAME}")
    plt.savefig(save_path / "avg_max_fitness.pdf", dpi=300, bbox_inches = "tight")
    plt.close()


def graph_max_max_fitness(problem: Problem,
                          parameters: Parameters,
                          results: list[RunLog],
                          save_path: Path):
    # Calculate the max max fitness
    max_max_fitness = np.zeros(len(results[0].best_fitness))
    for result in results:
        for i in range(len(result.best_fitness)):
            if result.best_fitness[i] > max_max_fitness[i]:
                max_max_fitness[i] = result.best_fitness[i]
    # Plot the max max fitness
    plt.plot(results[0].generation, max_max_fitness)
    # Plot the optimal fitness
    plt.plot(results[0].generation,
             np.full(len(results[0].generation),
             problem.get_max_fitness()),
             linestyle="--")
    plt.legend(["Max-Max Fitness", "Optimal Fitness"])
    plt.xlabel(f"Generation (Population Size: {parameters.population_size})")
    plt.ylabel("Max-Max Fitness")
    plt.title(f"Max-Max Fitness for {problem.NAME}")
    plt.savefig(save_path / "max_max_fitness.pdf", dpi=300, bbox_inches = "tight")
    plt.close()


def graph_avg_avg_tour_length(problem: Problem,
                              parameters: Parameters,
                              results: list[RunLog],
                              save_path: Path):
    # Calculate the average average tour length
    avg_avg_tour_length = np.zeros(len(results[0].avg_cost))
    for result in results:
        for i in range(len(result.avg_cost)):
            avg_avg_tour_length[i] += result.avg_cost[i]
    avg_avg_tour_length /= len(results)
    # Plot the average average tour length
    plt.plot(results[0].generation, avg_avg_tour_length)
    # Plot the optimal cost
    plt.plot(results[0].generation,
             np.full(len(results[0].generation),
             problem.get_min_cost()),
             linestyle="--")
    plt.legend(["Average-Average Tour Length", "Optimal Fitness"])
    plt.xlabel(f"Generation (Population Size: {parameters.population_size})")
    plt.ylabel("Average-Average Tour Length")
    plt.title(f"Average-Average Tour Length for {problem.NAME}")
    plt.savefig(save_path / "avg_avg_tour_length.pdf", dpi=300, bbox_inches = "tight")
    plt.close()


def graph_max_avg_tour_length(problem: Problem,
                              parameters: Parameters,
                              results: list[RunLog],
                              save_path: Path):
    # Calculate the max average tour length
    max_avg_tour_length = np.zeros(len(results[0].avg_cost))
    for result in results:
        for i in range(len(result.avg_cost)):
            if result.avg_cost[i] > max_avg_tour_length[i]:
                max_avg_tour_length[i] = result.avg_cost[i]
    # Plot the max average tour length
    plt.plot(results[0].generation, max_avg_tour_length)
    # Plot the optimal cost
    plt.plot(results[0].generation,
             np.full(len(results[0].generation),
             problem.get_min_cost()),
             linestyle="--")
    plt.legend(["Max-Average Tour Length", "Optimal Fitness"])
    plt.xlabel(f"Generation (Population Size: {parameters.population_size})")
    plt.ylabel("Max-Average Tour Length")
    plt.title(f"Max-Average Tour Length for {problem.NAME}")
    plt.savefig(save_path / "max_avg_tour_length.pdf", dpi=300, bbox_inches = "tight")
    plt.close()


def graph_avg_max_tour_length(problem: Problem,
                              parameters: Parameters,
                              results: list[RunLog],
                              save_path: Path):
    # Calculate the average max tour length
    avg_max_tour_length = np.zeros(len(results[0].avg_cost))
    for result in results:
        for i in range(len(result.best_cost)):
            avg_max_tour_length[i] += result.best_cost[i]
    avg_max_tour_length /= len(results)
    # Plot the average max tour length
    plt.plot(results[0].generation, avg_max_tour_length)
    # Plot the optimal cost
    plt.plot(results[0].generation,
             np.full(len(results[0].generation),
             problem.get_min_cost()),
             linestyle="--")
    plt.legend(["Average-Max Tour Length", "Optimal Fitness"])
    plt.xlabel(f"Generation (Population Size: {parameters.population_size})")
    plt.ylabel("Average-Max Tour Length")
    plt.title(f"Average-Max Tour Length for {problem.NAME}")
    plt.savefig(save_path / "avg_max_tour_length.pdf", dpi=300, bbox_inches = "tight")
    plt.close()


def graph_max_max_tour_length(problem: Problem,
                              parameters: Parameters,
                              results: list[RunLog],
                              save_path: Path):
    # Calculate the max max tour length
    max_max_tour_length = np.zeros(len(results[0].best_cost))
    for result in results:
        for i in range(len(result.best_cost)):
            if result.best_cost[i] > max_max_tour_length[i]:
                max_max_tour_length[i] = result.best_cost[i]
    # Plot the max max tour length
    plt.plot(results[0].generation, max_max_tour_length)
    # Plot the optimal cost
    plt.plot(results[0].generation,
             np.full(len(results[0].generation),
             problem.get_min_cost()),
             linestyle="--")
    plt.legend(["Max-Max Tour Length", "Optimal Fitness"])
    plt.xlabel(f"Generation (Population Size: {parameters.population_size})")
    plt.ylabel("Max-Max Tour Length")
    plt.title(f"Max-Max Tour Length for {problem.NAME}")
    plt.savefig(save_path / "max_max_tour_length.pdf", dpi=300, bbox_inches = "tight")
    plt.close()
