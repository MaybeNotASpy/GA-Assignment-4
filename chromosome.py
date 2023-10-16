from __future__ import annotations
import random
import deal
from math import ceil, log2
import numpy as np


class Chromosome():
    _genes: list[int] = []
    _bits_per_val: int = -1
    _num_of_genes: int

    def __init__(self, num_of_genes: int, bits_per_val: int = -1):
        """
        Args:
            num_of_genes (int): The size of the chromosome in genes.
            bits_per_val (int): The number of bits per value.
                Must be a positive integer.
                If -1, then bits_per_val is set to the smallest power of 2
                that is greater than or equal to size.
        """
        self._genes: list[int] = [0] * num_of_genes * bits_per_val
        if bits_per_val == -1 or bits_per_val < ceil(log2(num_of_genes)):
            bits_per_val = num_of_genes.bit_length()
        self._bits_per_val = bits_per_val
        self._num_of_genes = num_of_genes

    def __str__(self):
        return "".join([str(bit) for bit in self._genes])

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self._genes)

    def __getitem__(self, index):
        return self._genes[index]

    def __setitem__(self, index, value):
        self._genes[index] = value

    def set_chromosome(self, chromosome: list[int]):
        self._genes = chromosome

    def get_chromosome(self) -> list[int]:
        return self._genes

    def get_bits_per_val(self) -> int:
        return self._bits_per_val

    def set_bits_per_val(self, bits_per_val: int):
        self._bits_per_val = bits_per_val

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
        genes = [
            bin(i)[2:].zfill(self._bits_per_val)
            for i in range(self._num_of_genes)
        ]
        np.random.shuffle(genes)
        for i, gene in enumerate(genes):
            self.set_val(
                i,
                list(map(int, (list(gene)))))

    def is_valid(self) -> bool:
        """
        Check if the chromosome is a valid set.

        Returns:
            bool: True if the chromosome is a valid set.
        """
        list_of_vals = []
        for i in range(self._num_of_genes):
            val = self.get_val(i)
            if val in list_of_vals or -1 in val:
                return False
            list_of_vals.append(val)
        return True

    @deal.pre(lambda self, index: index < self._num_of_genes)
    def get_val(self, index: int) -> list[int]:
        """
        Get the slice of bits_per_val bits at index.

        Args:
            index (int): The index of the value to get.
                Will be multiplied by bits_per_val.

        Returns:
            list[int]: The slice of bits_per_val bits at index.
        """
        index *= self._bits_per_val
        return self._genes[index:index + self._bits_per_val]

    @deal.pre(lambda self, index, value: len(value) == self._bits_per_val)
    @deal.pre(lambda self, index, value: index < self._num_of_genes)
    @deal.pre(lambda self, index, value: all(
        bit in [-1, 0, 1] for bit in value))
    def set_val(self, index: int, value: list[int]):
        """
        Set the slice of bits_per_val bits at index.

        Args:
            index (int): The index of the value to set.
                Will be multiplied by bits_per_val.
            value (list[int]): The slice of bits_per_val bits to set.
                Must be of length bits_per_val.
        """
        index *= self._bits_per_val
        self._genes[index:index + self._bits_per_val] = value

    @deal.pre(lambda self, value: len(value) == self._bits_per_val)
    def find_val(self, value: list[int]) -> int:
        """
        Find the index of the occurrence of value.

        Args:
            value (list[int]): The value to find.
                Must be of length bits_per_val.

        Returns:
            int: The index of the occurrence of value.
                Returns -1 if value is not found.
        """
        for i in range(self._num_of_genes):
            if self.get_val(i) == value:
                return i
        return -1

    def set_empty_gene(self, index: int):
        """
        Set the gene at index to all -1s.

        Args:
            index (int): The index of the gene to set.
        """
        self.set_val(index, [-1] * self._bits_per_val)

    @deal.pre(lambda self, index1, index2: index1 < index2)
    @deal.pre(lambda self, index1, index2: index2 < self._num_of_genes)
    def swap(self, index1: int, index2: int):
        """
        Swap two bits_per_val slices in the chromosome.

        Args:
            index1 (int): The index of the first slice.
                Will be multiplied by bits_per_val.
                Also must be less than index2.
            index2 (int): The index of the second slice.
                Will be multiplied by bits_per_val.
                Also must be greater than index1.
        """
        index1 *= self._bits_per_val
        index2 *= self._bits_per_val
        self._genes[index1:index1 + self._bits_per_val], \
            self._genes[index2:index2 + self._bits_per_val] = \
            self._genes[index2:index2 + self._bits_per_val], \
            self._genes[index1:index1 + self._bits_per_val]

    def inversion_mutation(self):
        """
        Inversion mutation
        """
        # Choose two random indices
        gene_indices = list(range(self._num_of_genes))
        np.random.shuffle(gene_indices)
        i, j = gene_indices[:2]
        # Ensure i < j
        if i > j:
            i, j = j, i
        # Reverse each bits_per_val slice
        # Example: 000001000 -> 000100000 for bits_per_val = 2, i = 1, j = 2
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
        np.random.shuffle(gene_indices)
        i, j = gene_indices[:2]
        # Ensure i < j
        if i > j:
            i, j = j, i
        # Swap the the two bits_per_val slices
        self.swap(i, j)

    def hamming_distance(self, other: Chromosome) -> int:
        """
        Calculate the hamming distance between two chromosomes.

        Args:
            other (Chromosome): The other chromosome.

        Returns:
            int: The hamming distance between the two chromosomes.
        """
        return sum(
            1 for i in range(len(self))
            if self[i] != other[i]
        )

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
        for i in range(0, self._num_of_genes, 2):
            val1 = self.get_val(i)
            val2 = self.get_val(i + 1)
            pos1 = other.find_val(val1)
            pos2 = other.find_val(val2)
            if pos1 + 1 == pos2:
                continue
            dis += 1
        # Also check the edges between the first and last nodes
        val1 = self.get_val(0)
        val2 = self.get_val(self._num_of_genes - 1)
        pos1 = other.find_val(val1)
        pos2 = other.find_val(val2)
        if not (pos1 == 0 and pos2 == self._num_of_genes - 1):
            dis += 1
        return dis
