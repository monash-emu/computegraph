from copy import deepcopy

import pytest

from computegraph import ComputeGraph
from .test_graph import get_pop_dict
import computegraph.options as options

default_drawing_options = deepcopy(options.drawing)


def setup_function():
    options.drawing = deepcopy(default_drawing_options)


def teardown_function():
    options.drawing = deepcopy(default_drawing_options)


def test_draw_graph_default():
    pop_dict = get_pop_dict()

    pop_graph = ComputeGraph(pop_dict)

    pop_graph.draw()


@pytest.mark.parametrize("backend", ["plotly", "matplotlib"])
@pytest.mark.parametrize("use_graphviz", [True, False])
def test_draw_graph_with_options(backend, use_graphviz):

    options.drawing["backend"] = backend
    options.drawing["use_graphviz"] = use_graphviz

    pop_dict = get_pop_dict()

    pop_graph = ComputeGraph(pop_dict)

    pop_graph.draw()
