import numpy as np

from computegraph.types import local, Function, param
from computegraph.graph import ComputeGraph
from computegraph.utils import get_nested_graph_dict

var = local
func = Function


def getitem(obj, index):
    return obj[index]


def get_pop_dict():
    def gen_data_dict():
        return {
            "AUS": np.array(
                (
                    10.0,
                    30.0,
                )
            ),
            "MYS": np.array((5.0, 9.0, 1.4)),
        }

    pop_dict = {}
    pop_dict["iso"] = param("iso")
    pop_dict["pop_scale"] = param("pop_scale")
    pop_dict["pop_df"] = Function(gen_data_dict)
    pop_dict["country_pop"] = Function(getitem, [var("pop_df"), var("iso")])
    pop_dict["pop_stats"] = Function(
        lambda s: {"min": s.min(), "sum": s.sum()}, [var("country_pop")]
    )
    pop_dict["pop_sum"] = Function(getitem, [var("pop_stats"), "sum"])
    pop_dict["norm_pop"] = Function(np.divide, [var("country_pop"), var("pop_sum")])
    pop_dict["out_pop"] = Function(np.multiply, [var("norm_pop"), var("pop_scale")])

    return pop_dict


def test_build_and_run_graph():

    pop_dict = get_pop_dict()

    pop_graph = ComputeGraph(pop_dict)

    pop_func = pop_graph.get_callable()

    parameters = {"iso": "AUS", "pop_scale": 10.0}

    results = pop_func(parameters=parameters)

    np.testing.assert_array_equal(results["out_pop"], np.array([2.5, 7.5]))


def test_nested_graph():

    pop_dict = get_pop_dict()

    nested_pop_dict = get_nested_graph_dict(pop_dict, "population", {"iso": "iso"})

    nested_pop_dict["out_pop_sum"] = Function(np.sum, [local("population.out_pop")])

    pop_graph = ComputeGraph(nested_pop_dict)

    params_flat = {"population.pop_scale": 10.0, "iso": "AUS"}
    params_nested = {"population": {"pop_scale": 10.0}, "iso": "AUS"}

    f_flat = pop_graph.get_callable(nested_params=False)
    results = f_flat(parameters=params_flat)

    np.testing.assert_array_equal(results["population.out_pop"], np.array([2.5, 7.5]))
    assert results["out_pop_sum"] == 10.0

    f_nested = pop_graph.get_callable(nested_params=True)
    results = f_nested(parameters=params_nested)

    np.testing.assert_array_equal(results["population.out_pop"], np.array([2.5, 7.5]))
    assert results["out_pop_sum"] == 10.0


def test_build_and_draw_graph():
    pop_dict = get_pop_dict()

    pop_graph = ComputeGraph(pop_dict)

    pop_graph.draw()
