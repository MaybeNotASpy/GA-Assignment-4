from individual import Individual
from coding import decode


class TestGenome():
    def test_mutate(self):
        for i in range(2, 100):
            genome = Individual(i)
            genome.mutate()
            assert genome.chromosome != \
                Individual(i).chromosome

    def test_crossover(self):
        for i in range(2, 100):
            genome1 = Individual(i)
            genome2 = Individual(i)
            genome1.mutate()
            genome2.mutate()
            child = genome1.crossover(genome2)
            # Assert that it is still a set of unique values
            tour = decode(child.chromosome)
            assert len(tour) == len(set(tour))
