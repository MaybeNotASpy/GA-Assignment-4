from chromosome import Chromosome
from node import Tour


def encode(nodes: Tour) -> Chromosome:
    """
    This function is used to encode the data
    from a set of nodes into a chromosome.

    Args:
        nodes (list[int]): The set of nodes to encode.

    Returns:
        Chromosome: The chromosome containing the encoded data.
    """
    chromosome = Chromosome(len(nodes))
    chromosome.chromosome = [node - 1 for node in nodes]

    return chromosome


def decode(chromosome: Chromosome) -> Tour:
    """
    This function is used to decode the data
    from a chromosome into a set of nodes.

    Args:
        chromosome (Chromosome): The chromosome to decode.

    Returns:
        Tour: The set of nodes decoded from the chromosome.
    """
    nodes = [node + 1 for node in chromosome.chromosome]

    return nodes
