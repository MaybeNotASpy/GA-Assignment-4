from __future__ import annotations
import numpy as np
from chromosome import Chromosome
import deal

generator = np.random.default_rng()


class Individual():
    """
    An individual in the population.

    Args:
        num_of_genes (int): The number of genes in the chromosome.
    """
    def __init__(self, num_of_genes: int):
        self._chromosome = Chromosome(num_of_genes)
        self._fitness = -1
        self._cost = -1
        self._chromosome.initialize()

    @property
    def num_of_genes(self) -> int:
        return self._chromosome.num_of_genes

    def __str__(self):
        return str(self._chromosome)

    def __len__(self):
        return len(self._chromosome)

    def initialize(self):
        self._chromosome.initialize()

    @property
    def chromosome(self) -> Chromosome:
        return self._chromosome

    @chromosome.setter
    def chromosome(self, chromosome: Chromosome):
        self._chromosome = chromosome

    @property
    def fitness(self) -> float:
        return self._fitness

    @fitness.setter
    def fitness(self, fitness: float):
        self._fitness = fitness

    @property
    def cost(self) -> float:
        return self._cost

    @cost.setter
    def cost(self, cost: float):
        self._cost = cost

    def is_valid(self) -> bool:
        return self._chromosome.is_valid()

    def mutate(self, mutation_rate: float = 0.1):
        """
        Mutate the chromosome.
        Random chance of using inversion or swap mutation.

        Args:
            mutation_rate (float): The probability of mutation per gene.
        """
        mutations = generator.binomial(
            self.num_of_genes,
            mutation_rate
        )
        for _ in range(mutations):
            if np.random.rand() < 0.5:
                self._chromosome.inversion_mutation()
            else:
                self._chromosome.swap_mutation()

    @deal.pre(lambda self, other: self.num_of_genes == other.num_of_genes)
    @deal.pre(lambda self, other: self.is_valid() and other.is_valid())
    @deal.post(lambda result: result.is_valid())
    def crossover(self, other: Individual):
        """
        PMX crossover

        Args:
            self: individual
            other: individual

        Returns:
            child: individual
        """
        child = Individual(self.num_of_genes)
        for i in range(self.num_of_genes):
            child._chromosome.set_empty_gene(i)
        filled_bools = [False] * self.num_of_genes

        # Get the crossover points. Make sure that the length of the
        # crossover segment is less than the number of genes.
        i, j = 0, 0
        while i == j or j - i > self.num_of_genes:
            indices = list(range(self.num_of_genes))
            generator.shuffle(indices)
            i, j = indices[:2]
            if i > j:
                i, j = j, i
            j += 1  # Add 1 to include last index

        # Copy the crossover segment from parent 0 to child
        for it in range(i, j):
            child._chromosome[it] = self._chromosome[it]
            filled_bools[it] = True

        # For each current gene in child, find the corresponding location in
        # parent 1 and copy the gene in the original position in parent 1 to
        # the child in the corresponding location.
        for it in range(i, j):
            if all(filled_bools):
                break
            parent_val = other._chromosome[it]
            if child._chromosome.find_val(parent_val) != -1:
                continue
            corr_location = other._chromosome.find_val(parent_val)
            while filled_bools[corr_location]:
                corr_location = other._chromosome.find_val(
                    child._chromosome[corr_location]
                )
            child._chromosome[corr_location] = parent_val
            filled_bools[corr_location] = True

        # Fil in the remaining genes in child with the genes from parent 1
        # in order of appearance.
        parent_it = 0
        for child_it in range(self.num_of_genes):
            if not filled_bools[child_it]:
                unique_val = other._chromosome[parent_it]
                while child._chromosome.find_val(unique_val) != -1:
                    parent_it += 1
                    unique_val = other._chromosome[parent_it]
                child._chromosome[child_it] = unique_val
                filled_bools[child_it] = True

        return child
