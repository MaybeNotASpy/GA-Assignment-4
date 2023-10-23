from pathlib import Path
from GA import RunLog
from problem import Problem
from parameters import Parameters
from coding import decode


def save_final_runs(problem: Problem,
                    parameters: Parameters,
                    results: list[RunLog],
                    save_path: Path) -> None:
    save_path.mkdir(parents=True, exist_ok=True)
    for run_result in results:
        run_save_path = save_path / f"run_{run_result.run_number}"
        run_save_path.mkdir(parents=True, exist_ok=True)
        # Save the fitness per generation
        with open(run_save_path / "fitness.csv", "w") as file:
            file.write("generation,avg_fitness,max_fitness\n")
            for i in range(len(run_result.avg_fitness)):
                file.write(f"{run_result.generation[i]},\
                           {run_result.avg_fitness[i]},\
                           {run_result.best_fitness[i]}\n")
        # Save the tour length per generation
        with open(run_save_path / "tour_length.csv", "w") as file:
            file.write("generation,avg_tour_length,best_tour_length\n")
            for i in range(len(run_result.avg_cost)):
                file.write(f"{run_result.generation[i]},\
                           {run_result.avg_cost[i]},\
                           {run_result.best_cost[i]}\n")
        # Save the evaluations per generation
        with open(run_save_path / "evaluations.csv", "w") as file:
            file.write("generation,evaluations\n")
            for i in range(len(run_result.evals)):
                file.write(f"{run_result.generation[i]},\
                           {run_result.evals[i]}\n")
    # Save the generations to reach best fitness per run
    with open(save_path / "gens_to_reach_best.csv", "w") as file:
        file.write("run_number,gens_to_reach_best\n")
        for run_result in results:
            file.write(f"{run_result.run_number},\
                       {run_result.final_gen}\n")
    # Save the best fitness and tour length per run
    with open(save_path / "best_fitness.csv", "w") as file:
        file.write("run_number,best_fitness\n")
        for run_result in results:
            file.write(f"{run_result.run_number},\
                       {run_result.best.fitness}\n")
    with open(save_path / "best_tour_length.csv", "w") as file:
        file.write("run_number,best_tour_length\n")
        for run_result in results:
            file.write(f"{run_result.run_number},\
                       {run_result.best.cost}\n")
    # Save the avg fitness and tour length per run
    with open(save_path / "avg_fitness.csv", "w") as file:
        file.write("run_number,avg_fitness\n")
        for run_result in results:
            file.write(f"{run_result.run_number},\
                       {run_result.avg_fitness[-1]}\n")
    with open(save_path / "avg_tour_length.csv", "w") as file:
        file.write("run_number,avg_tour_length\n")
        for run_result in results:
            file.write(f"{run_result.run_number},\
                       {run_result.avg_cost[-1]}\n")
    # Save the percentage of optimal fitness and multiple of optimal tour length per run  # noqa
    with open(save_path / "quality.csv", "w") as file:
        file.write("run_number,perc_of_optimal_fitness,multiple_of_optimal_tour_length\n") # noqa
        for run_result in results:
            file.write(f"{run_result.run_number},\
                       {run_result.best.fitness / problem.get_max_fitness()},\
                       {run_result.best.cost / problem.get_min_cost()}\n")
    # Save the number of evaluations per run that are within some percentage of the optimal fitness  # noqa
    with open(save_path / "evaluations_within_perc_of_optimal.csv", "w") as file:  # noqa
        file.write("run_number,1_perc,2_perc,5_perc,10_perc,20_perc\n")
        # If run doesn't reach perc, put -1
        for run_result in results:
            one_perc = -1
            two_perc = -1
            five_perc = -1
            ten_perc = -1
            twenty_perc = -1
            for i in range(len(run_result.evals)):
                if run_result.best_cost[i] >= problem.SOLUTION_COST * 0.99:
                    one_perc = run_result.evals[i]
                if run_result.best_cost[i] >= problem.SOLUTION_COST * 0.98:
                    two_perc = run_result.evals[i]
                if run_result.best_cost[i] >= problem.SOLUTION_COST * 0.95:
                    five_perc = run_result.evals[i]
                if run_result.best_cost[i] >= problem.SOLUTION_COST * 0.90:
                    ten_perc = run_result.evals[i]
                if run_result.best_cost[i] >= problem.SOLUTION_COST * 0.80:
                    twenty_perc = run_result.evals[i]
                    break
            file.write(f"{run_result.run_number},\
                        {one_perc},\
                        {two_perc},\
                        {five_perc},\
                        {ten_perc},\
                        {twenty_perc}\n")
    # Save the parameters
    with open(save_path / "parameters.csv", "w") as file:
        file.write("population_size,max_generations,mutation_rate,perc_default_min_distance\n")  # noqa
        file.write(f"{parameters.population_size},\
                   {parameters.max_generations},\
                   {parameters.mutation_rate},\
                   {parameters.perc_default_min_distance}\n")
    # Save the best tour across all runs
    best = results[0].best
    for run_result in results:
        if run_result.best.fitness > best.fitness:
            best = run_result.best
    with open(save_path / "best_tour.csv", "w") as file:
        file.write("node\n")
        tour = decode(best.chromosome)
        for node in tour:
            file.write(f"{node}\n")
