from __future__ import annotations
import deal
import numpy as np

generator = np.random.default_rng()


class Chromosome():
    def __init__(self, num_of_genes: int):
        """
        Args:
            num_of_genes (int): The size of the chromosome in genes.
        """
        self._genes: list[int] = [i for i in range(num_of_genes)]
        self._num_of_genes = num_of_genes

    def __str__(self):
        return str(self._genes)

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self._genes)

    def __getitem__(self, index) -> int:
        return self._genes[index]

    def __setitem__(self, index, value):
        self._genes[index] = value

    @property
    def chromosome(self) -> list[int]:
        return self._genes

    @chromosome.setter
    def chromosome(self, chromosome: list[int]):
        self._genes = chromosome
        self._num_of_genes = len(chromosome)

    @property
    def num_of_genes(self) -> int:
        return self._num_of_genes

    def __iter__(self):
        return iter(self._genes)

    def __eq__(self, other: Chromosome):
        return self._genes == other._genes

    def __ne__(self, other: Chromosome):
        return self._genes != other._genes

    def initialize(self):
        """
        Initialize the chromosome with a set of random genes.
        """
        generator.shuffle(self._genes)

    def is_valid(self) -> bool:
        """
        Check if the chromosome is a valid set.

        Returns:
            bool: True if the chromosome is a valid set.
        """
        return len(self._genes) == len(set(self._genes))

    def find_val(self, value: int) -> int:
        """
        Find the index of the occurrence of value.

        Args:
            value (int): The value to find.

        Returns:
            int: The index of the occurrence of value.
                Returns -1 if value is not found.
        """
        try:
            return self._genes.index(value)
        except ValueError:
            return -1

    def set_empty_gene(self, index: int):
        """
        Set the gene at index to -1.

        Args:
            index (int): The index of the gene to set.
        """
        self._genes[index] = -1

    @deal.pre(lambda self, index1, index2: index1 < index2)
    @deal.pre(lambda self, index1, index2: index2 < self._num_of_genes)
    @deal.pre(lambda self, index1, index2: index1 >= 0)
    def swap(self, index1: int, index2: int):
        """
        Swap two genes in the chromosome.

        Args:
            index1 (int): The index of the first gene.
            index2 (int): The index of the second gene.
        """
        self._genes[index1], self._genes[index2] = \
            self._genes[index2], self._genes[index1]

    def inversion_mutation(self):
        """
        Inversion mutation
        """
        # Choose two random indices
        gene_indices = list(range(self._num_of_genes))
        generator.shuffle(gene_indices)
        i, j = gene_indices[:2]
        # Ensure i < j
        if i > j:
            i, j = j, i
        # Swap each gene around the middle
        # Example: 1 2 3 4 5 -> 5 4 3 2 1
        while i < j:
            self.swap(i, j)
            i += 1
            j -= 1

    def swap_mutation(self):
        """
        Swap mutation
        """
        # Choose two random indices
        gene_indices = list(range(self._num_of_genes))
        generator.shuffle(gene_indices)
        i, j = gene_indices[:2]
        # Ensure i < j
        if i > j:
            i, j = j, i
        # Swap the the two genes
        self.swap(i, j)

    def edgewise_distance(self, other: Chromosome) -> int:
        """
        Calculate the edgewise distance between two chromosomes.
        This is the number of edges that are different between the two
        chromosomes. Order of the edges does not matter.

        Args:
            other (Chromosome): The other chromosome.

        Returns:
            int: The edgewise distance between the two chromosomes.
        """
        dis = 0
        for i in range(0, self._num_of_genes - 1):
            val1 = self[i]
            val2 = self[i + 1]
            pos1 = other.find_val(val1)
            pos2 = other.find_val(val2)
            if pos1 + 1 == pos2:
                continue
            dis += 1
        # Also check the edges between the first and last nodes
        val1 = self[0]
        val2 = self[-1]
        pos1 = other.find_val(val1)
        pos2 = other.find_val(val2)
        if not (pos1 == 0 and pos2 == self._num_of_genes - 1):
            dis += 1
        return dis
