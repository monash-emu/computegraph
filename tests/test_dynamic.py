from computegraph.types import Function, param, local
from computegraph.utils import get_nested_graph_dict

from .test_graph import get_pop_dict


def _test_split_graph():

    pop_dict = get_pop_dict()

    targets = ["out_pop", "pop_sum"]
    dynamic_inputs = [param("pop_scale")]

    parameters = {"pop_scale": 10.0, "iso": "AUS"}

    static_cg = split_static_dynamic(pop_dict, dynamic_inputs, targets, parameters=parameters)

    # Check that the graph contains all our targets
    assert set(targets).issubset(set(static_cg.dict.keys()))

    # Check that we only require the inputs that we specified above
    assert static_cg.get_input_variables() == set(dynamic_inputs)

    # Run the graph
    out_dict = static_cg.get_callable()(parameters=parameters)

    # Check for expected outputs
    assert out_dict["pop_sum"] == 40.0
