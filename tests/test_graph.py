import pytest
import numpy as np


from computegraph.types import Variable, local, Function, param
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
    pop_dict["pop_df"] = Function(gen_data_dict)
    pop_dict["country_pop"] = Function(getitem, [var("pop_df"), param("iso")])
    pop_dict["pop_stats"] = Function(
        lambda s: {"min": s.min(), "sum": s.sum()}, [var("country_pop")]
    )
    pop_dict["pop_sum"] = Function(getitem, [var("pop_stats"), "sum"])
    pop_dict["norm_pop"] = Function(np.divide, [var("country_pop"), var("pop_sum")])
    pop_dict["out_pop"] = Function(np.multiply, [var("norm_pop"), param("pop_scale")])

    return pop_dict


def test_build_and_run_graph():

    pop_dict = get_pop_dict()

    pop_graph = ComputeGraph(pop_dict, is_traced=True)

    pop_func = pop_graph.get_callable(output_all=True)

    parameters = {"iso": "AUS", "pop_scale": 10.0}

    results = pop_func(parameters=parameters)

    np.testing.assert_array_equal(results["out_pop"], np.array([2.5, 7.5]))


def get_simple_jax_dict():
    jdict = {}
    jdict["out"] = Function(lambda x, y: x * y, [param("x"), param("y")])
    return jdict


def test_jax_jit():
    try:
        from jax import jit
    except:
        return

    jdict = get_simple_jax_dict()

    jgraph = ComputeGraph(jdict)
    fcall = jit(jgraph.get_callable())

    results = fcall(parameters={"x": 2.0, "y": 5.0})

    assert results["out"] == 10.0
