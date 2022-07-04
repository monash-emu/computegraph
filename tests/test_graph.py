import numpy as np

from computegraph.types import local, Function, param
from computegraph.graph import ComputeGraph

var = local
func = Function


def getitem(obj, index):
    return obj[index]


def get_pop_dict():
    def gen_data_dict():
        return {"AUS": np.array((12.1, 92.1, 2.1)), "MYS": np.array((5.0, 9.0, 1.4))}

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

    parameters = {"iso": "AUS", "pop_scale": 0.1}

    pop_func(parameters=parameters)


def test_build_and_draw_graph():
    pop_dict = get_pop_dict()

    pop_graph = ComputeGraph(pop_dict)

    pop_graph.draw()
