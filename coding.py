from chromosome import Chromosome
from node import Node, Tour


def encode(nodes: Tour, bits_per_val: int = -1) -> Chromosome:
    """
    This function is used to encode the data
    from a set of nodes into a chromosome.

    Args:
        nodes (set[tuple[int, int]]): The set of nodes to encode.
        bits_per_val (int, optional): The number of bits to use per value.
            Defaults to -1.

    Returns:
        Chromosome: The chromosome containing the encoded data.
    """
    if bits_per_val == -1:
        bits_per_val = len(nodes).bit_length()

    chromosome = Chromosome(len(nodes), bits_per_val)

    for i, node in enumerate(nodes):
        binary = bin(node)[2:]  # Remove the 0b prefix
        binary = binary.zfill(bits_per_val)  # Pad with 0s
        binary = list(map(int, list(binary)))  # Convert to list of ints
        chromosome.set_val(i, binary)

    return chromosome


def decode(chromosome: Chromosome) -> Tour:
    """
    This function is used to decode the data
    from a chromosome into a set of nodes.

    Args:
        chromosome (Chromosome): The chromosome to decode.
        bits_per_val (int): The number of bits to use per value.

    Returns:
        set[tuple[int, int]]: The set of nodes decoded from the chromosome.
    """
    bits_per_val = chromosome.get_bits_per_val()
    nodes = Tour([None] * (len(chromosome) // bits_per_val))

    for i in range(len(chromosome) // bits_per_val):
        slice = chromosome.get_val(i)
        nodes[i] = int("".join([str(bit) for bit in slice]), 2)
    return nodes
