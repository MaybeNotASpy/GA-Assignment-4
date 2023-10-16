from individual import Individual
from coding import decode
from math import ceil, log2


class TestGenome():
    def test_mutate(self):
        for i in range(2, 100):
            bits = ceil(log2(i))
            genome = Individual(i, bits)
            genome.mutate()
            assert genome.get_chromosome() != \
                Individual(i, bits).get_chromosome()

    def test_crossover(self):
        for i in range(2, 100):
            bits = ceil(log2(i))
            genome1 = Individual(i, bits)
            genome2 = Individual(i, bits)
            genome1.mutate()
            genome2.mutate()
            child = genome1.crossover(genome2)
            # Assert that it is still a set of unique values
            tour = decode(child.get_chromosome())
            assert len(tour) == len(set(tour))
