from coding import encode, decode
from node import Tour


class TestCoding():
    def test_encode(self):
        for i in range(1, 100):
            raw_set = set(range(i))
            nodes = Tour(raw_set)
            bits_per_val = len(raw_set).bit_length()
            chromosome = encode(nodes, bits_per_val)
            assert chromosome.get_bits_per_val() == bits_per_val
            assert len(chromosome) == len(nodes) * bits_per_val

    def test_decode(self):
        for i in range(2, 100):
            raw_set = list(range(1, i))
            nodes = Tour(raw_set)
            bits_per_val = len(raw_set).bit_length()
            chromosome = encode(nodes, bits_per_val)
            decoded = decode(chromosome)
            print(decoded)
            print(nodes)
            assert nodes == decoded
