from __future__ import annotations
import numpy as np
from chromosome import Chromosome
import random
import deal

generator = np.random.default_rng()


class Individual():
    def __init__(self, num_of_genes: int, bits_per_val: int = -1):
        self._chromosome = Chromosome(num_of_genes, bits_per_val)
        self._num_of_genes = num_of_genes
        self._bits_per_val = bits_per_val
        if bits_per_val == -1:
            self._bits_per_val = num_of_genes.bit_length()
        self._fitness = -1
        self._cost = -1
        self._chromosome.initialize()

    def __str__(self):
        return str(self._chromosome)

    def __len__(self):
        return len(self._chromosome)

    def initialize(self):
        self._chromosome.initialize()

    def get_bits_per_val(self) -> int:
        return self._bits_per_val

    def get_number_of_genes(self) -> int:
        return self._num_of_genes

    def get_chromosome(self) -> Chromosome:
        return self._chromosome

    def get_fitness(self) -> float:
        return self._fitness

    def set_fitness(self, fitness: float):
        self._fitness = fitness

    def get_cost(self) -> float:
        return self._cost

    def set_cost(self, cost: float):
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
            len(self._chromosome) // self._bits_per_val,
            mutation_rate
        )
        for _ in range(mutations):
            if np.random.rand() < 0.5:
                self._chromosome.inversion_mutation()
            else:
                self._chromosome.swap_mutation()

    @deal.pre(lambda self, other: self._num_of_genes == other._num_of_genes)
    @deal.pre(lambda self, other: self._bits_per_val == other._bits_per_val)
    @deal.pre(lambda self, other: self.is_valid() and other.is_valid())
    def crossover(self, other: Individual):
        """
        PMX crossover

        Args:
            self: individual
            other: individual

        Returns:
            child: individual
        """
        child = Individual(self._num_of_genes, self._bits_per_val)
        for i in range(self._num_of_genes):
            child._chromosome.set_empty_gene(i)
        filled_bools = [False] * self._num_of_genes

        # Get the crossover points. Make sure that the length of the
        # crossover segment is less than the number of genes.
        i, j = 0, 0
        while i == j or j - i > self._num_of_genes:
            i, j = random.sample(
                range(self._num_of_genes),
                2
            )
            if i > j:
                i, j = j, i
            j += 1  # Add 1 to include last index

        # Copy the crossover segment from parent 0 to child
        for it in range(i, j):
            child._chromosome.set_val(
                it,
                self._chromosome.get_val(it))
            filled_bools[it] = True

        # For each current gene in child, find the corresponding location in
        # parent 1 and copy the gene in the original position in parent 1 to
        # the child in the corresponding location.
        for it in range(i, j):
            if all(filled_bools):
                break
            parent_val = other._chromosome.get_val(it)
            if child._chromosome.find_val(parent_val) != -1:
                continue
            corr_location = other._chromosome.find_val(parent_val)
            while filled_bools[corr_location]:
                corr_location = other._chromosome.find_val(
                    child._chromosome.get_val(corr_location)
                )
            child._chromosome.set_val(
                corr_location,
                parent_val)
            filled_bools[corr_location] = True

        # Fil in the remaining genes in child with the genes from parent 1
        # in order of appearance.
        parent_it = 0
        for child_it in range(self._num_of_genes):
            if not filled_bools[child_it]:
                unique_val = other._chromosome.get_val(parent_it)
                while child._chromosome.find_val(unique_val) != -1:
                    parent_it += 1
                    unique_val = other._chromosome.get_val(parent_it)
                child._chromosome.set_val(child_it, unique_val)
                filled_bools[child_it] = True

        assert child.is_valid()
        return child
