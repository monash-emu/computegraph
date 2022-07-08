from computegraph.types import Function, param, local
from computegraph.dynamic import split_static_dynamic
from computegraph.utils import get_nested_graph_dict

from .test_graph import get_pop_dict


def test_split_graph():

    pop_dict = get_pop_dict()

    targets = ["out_pop"]
    dynamic_inputs = [param("pop_scale")]

    parameters = {"pop_scale": 10.0, "iso": "AUS"}

    split_static_dynamic(pop_dict, dynamic_inputs, targets, parameters=parameters)


def test_split_nested():

    pop_dict = get_pop_dict()

    dynamic_inputs = [param("population.pop_scale")]

    nested_pop_dict = get_nested_graph_dict(
        pop_dict, "population", True, param_map={"parameters": {"iso": "iso"}}
    )
    nested_pop_dict["out_pop_sum"] = Function(sum, [local("population.out_pop")])

    targets = ["out_pop_sum"]
    params_nested = {"population": {"pop_scale": 10.0}, "iso": "AUS"}

    split_static_dynamic(nested_pop_dict, dynamic_inputs, targets, parameters=params_nested)
