from coding import encode, decode
from node import Tour


class TestCoding():
    def test_encode(self):
        for i in range(1, 100):
            raw_set = set(range(i))
            nodes = Tour(raw_set)
            chromosome = encode(nodes)
            assert len(chromosome) == len(nodes)

    def test_decode(self):
        for i in range(2, 100):
            raw_set = list(range(1, i))
            nodes = Tour(raw_set)
            chromosome = encode(nodes)
            decoded = decode(chromosome)
            print(decoded)
            print(nodes)
            assert nodes == decoded
