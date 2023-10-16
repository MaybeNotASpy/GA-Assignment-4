

Node = int


class Tour():
    def __init__(self, tour: list[Node] = []):
        self._tour = tour

    def __len__(self):
        return len(self._tour)

    def __str__(self):
        return str(self._tour)

    def __getitem__(self, index):
        return self._tour[index]

    def __setitem__(self, index, value):
        self._tour[index] = value

    def __iter__(self):
        return iter(self._tour)

    def __eq__(self, other):
        return self._tour == other._tour

    def append(self, node: Node):
        self._tour.append(node)
