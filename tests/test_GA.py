from GA import GA
from individual import Individual
from load_tsp import get_tsp_data, get_tour_data


class TestGA:
    def test_evaluate(self):
        ga = GA(perc_default_min_distance=0.1)
        problem = get_tsp_data("data/berlin52.tsp")
        problem_solution = get_tour_data("data/berlin52.opt.tour")
        problem_solution.load_adjacency_matrix(problem.ADJACENCY_MATRIX)
        problem.SOLUTION_COST = problem_solution.get_distance()
        population = [Individual(52) for _ in range(100)]
        ga.evaluate(population, problem)
        for individual in population:
            assert individual.fitness is not None

    def test_crossover(self):
        ga = GA(perc_default_min_distance=0.1)
        population = [Individual(52) for _ in range(100)]
        for individual in population:
            individual.mutate()
        children = ga.crossover(population)
        assert len(children) == len(population) // 2

    def test_mutate(self):
        ga = GA(perc_default_min_distance=0.1)
        population = [Individual(52) for _ in range(100)]
        mutated_population = ga.mutate(population)
        assert len(mutated_population) == len(population)
