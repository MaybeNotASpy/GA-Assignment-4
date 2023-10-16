from distances import (
    manhattan_distance,
    maximum_distance,
    convert_to_lat_long,
    pseudo_euclidean_distance,
    ceiling_of_euclidean_distance
)
from load_tsp import get_tsp_data, get_tour_data, EDGE_WEIGHT_TYPE
from numpy import isclose


class TestDistances():
    def test_euclidean_distance(self):
        berlin52 = get_tsp_data("data/berlin52.tsp")
        berlin52_solution = get_tour_data("data/berlin52.opt.tour")
        berlin52_solution._edge_weight_type = EDGE_WEIGHT_TYPE.EUC_2D
        berlin52_solution._adjacency_matrix = berlin52.ADJACENCY_MATRIX
        assert berlin52_solution.get_distance() == 7542

    def test_manhattan_distance(self):
        assert manhattan_distance((0, 0), (0, 10)) == 10
        assert manhattan_distance((0, 0), (10, 0)) == 10
        assert manhattan_distance((0, 0), (10, 10)) == 20
        assert manhattan_distance((0, 10), (0, 0)) == 10
        assert manhattan_distance((0, 10), (0, 10)) == 0
        assert manhattan_distance((0, 10), (10, 0)) == 20
        assert manhattan_distance((0, 10), (10, 10)) == 10
        assert manhattan_distance((10, 0), (0, 0)) == 10
        assert manhattan_distance((10, 0), (0, 10)) == 20
        assert manhattan_distance((10, 0), (10, 0)) == 0
        assert manhattan_distance((10, 0), (10, 10)) == 10
        assert manhattan_distance((10, 10), (0, 0)) == 20
        assert manhattan_distance((10, 10), (0, 10)) == 10
        assert manhattan_distance((10, 10), (10, 0)) == 10
        assert manhattan_distance((10, 10), (10, 10)) == 0

    def test_maximum_distance(self):
        assert maximum_distance((0, 0), (0, 10)) == 10
        assert maximum_distance((0, 0), (10, 0)) == 10
        assert maximum_distance((0, 0), (10, 10)) == 10
        assert maximum_distance((0, 10), (0, 0)) == 10
        assert maximum_distance((0, 10), (0, 10)) == 0
        assert maximum_distance((0, 10), (10, 0)) == 10
        assert maximum_distance((0, 10), (10, 10)) == 10
        assert maximum_distance((10, 0), (0, 0)) == 10
        assert maximum_distance((10, 0), (0, 10)) == 10
        assert maximum_distance((10, 0), (10, 0)) == 0
        assert maximum_distance((10, 0), (10, 10)) == 10
        assert maximum_distance((10, 10), (0, 0)) == 10
        assert maximum_distance((10, 10), (0, 10)) == 10
        assert maximum_distance((10, 10), (10, 0)) == 10
        assert maximum_distance((10, 10), (10, 10)) == 0

    def test_convert_to_lat_long(self):
        # Should be within 7 significant digits of the true value
        test = convert_to_lat_long((0, 0))
        target = (0.0, 0.0)
        assert isclose(test[0], target[0], atol=1e-5)
        test = convert_to_lat_long((0, 10))
        target = (0.0, 0.17453292519943295)
        assert isclose(test[0], target[0], atol=1e-5)
        test = convert_to_lat_long((10, 0))
        target = (0.17453292519943295, 0.0)
        assert isclose(test[0], target[0], atol=1e-5)
        test = convert_to_lat_long((10, 10))
        target = (0.17453292519943295, 0.17453292519943295)
        assert isclose(test[0], target[0], atol=1e-5)

    def test_geographical_distance(self):
        ulysses16 = get_tsp_data("data/ulysses16.tsp")
        ulysses16_solution = get_tour_data("data/ulysses16.opt.tour")
        ulysses16_solution._edge_weight_type = EDGE_WEIGHT_TYPE.GEO
        ulysses16_solution._adjacency_matrix = ulysses16.ADJACENCY_MATRIX
        assert ulysses16_solution.get_distance() == 6859

    def test_pseudo_euclidean_distance(self):
        assert pseudo_euclidean_distance((0, 0), (0, 10)) == 4
        assert pseudo_euclidean_distance((0, 0), (10, 0)) == 4
        assert pseudo_euclidean_distance((0, 0), (10, 10)) == 5
        assert pseudo_euclidean_distance((0, 10), (0, 0)) == 4
        assert pseudo_euclidean_distance((0, 10), (0, 10)) == 0
        assert pseudo_euclidean_distance((0, 10), (10, 0)) == 5
        assert pseudo_euclidean_distance((0, 10), (10, 10)) == 4
        assert pseudo_euclidean_distance((10, 0), (0, 0)) == 4
        assert pseudo_euclidean_distance((10, 0), (0, 10)) == 5
        assert pseudo_euclidean_distance((10, 0), (10, 0)) == 0
        assert pseudo_euclidean_distance((10, 0), (10, 10)) == 4
        assert pseudo_euclidean_distance((10, 10), (0, 0)) == 5
        assert pseudo_euclidean_distance((10, 10), (0, 10)) == 4
        assert pseudo_euclidean_distance((10, 10), (10, 0)) == 4
        assert pseudo_euclidean_distance((10, 10), (10, 10)) == 0

    def test_ceiling_of_euclidean_distance(self):
        assert ceiling_of_euclidean_distance((0, 0), (0, 10)) == 10
        assert ceiling_of_euclidean_distance((0, 0), (10, 0)) == 10
        assert ceiling_of_euclidean_distance((0, 0), (10, 10)) == 15
        assert ceiling_of_euclidean_distance((0, 10), (0, 0)) == 10
        assert ceiling_of_euclidean_distance((0, 10), (0, 10)) == 0
        assert ceiling_of_euclidean_distance((0, 10), (10, 0)) == 15
        assert ceiling_of_euclidean_distance((0, 10), (10, 10)) == 10
        assert ceiling_of_euclidean_distance((10, 0), (0, 0)) == 10
        assert ceiling_of_euclidean_distance((10, 0), (0, 10)) == 15
        assert ceiling_of_euclidean_distance((10, 0), (10, 0)) == 0
        assert ceiling_of_euclidean_distance((10, 0), (10, 10)) == 10
        assert ceiling_of_euclidean_distance((10, 10), (0, 0)) == 15
        assert ceiling_of_euclidean_distance((10, 10), (0, 10)) == 10
        assert ceiling_of_euclidean_distance((10, 10), (10, 0)) == 10
        assert ceiling_of_euclidean_distance((10, 10), (10, 10)) == 0
