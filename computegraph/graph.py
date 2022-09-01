from typing import Callable, Any, List

import networkx as nx

from .types import Variable, Function, Data, GraphDict, GraphObject
from .draw import draw_compute_graph
from .utils import trace_with_named_keys, trace_object, get_input_variables, filter_graph, query

"""
Graph building functions
"""


def build_dag(graph_dict: dict, local_source_name="graph_locals") -> nx.DiGraph:
    """Build a DiGraph from the supplied GraphDict

    Args:
        graph_dict (GraphDict): _description_

    Returns:
        nx.DiGraph: A fully traced computegraph DiGraph
    """
    dag = nx.digraph.DiGraph()
    for name, node_spec in graph_dict.items():
        if not (isinstance(node_spec, GraphObject)):
            raise TypeError(
                "Invalid node type, expected Function, Variable, or Data", type(node_spec)
            )
        dag.add_node(name, node_spec=node_spec)
    trace_edges(dag, local_source_name)
    return dag


def trace_edges(dag: nx.DiGraph, local_source_name="graph_locals"):
    """Trace the DAG and create the appropriate edges (in-place)

    Args:
        dag: A DiGraph whose keys are names, with attribute 'node_spec'
    """

    def find_source_node(dag: nx.DiGraph, var: Variable):
        source_key = var.key
        source_node = None
        if source_key in dag:
            source_node = source_key
        else:
            raise KeyError(source_key)
        return source_node

    for node in dag:
        node_spec = dag.nodes[node]["node_spec"]
        if isinstance(node_spec, Function):
            for a in node_spec.args:
                if isinstance(a, Variable) and (a.source == local_source_name):
                    source_node = find_source_node(dag, a)
                    dag.add_edge(source_node, node)
            for k, v in node_spec.kwargs.items():
                if isinstance(v, Variable) and (v.source == local_source_name):
                    # We have a nested key
                    source_node = find_source_node(dag, v)
                    dag.add_edge(source_node, node)


"""
Operations over built DiGraphs
"""


def build_callable(
    dag: nx.DiGraph,
    targets=None,
    include_inputs=False,
    local_source_name="graph_locals",
) -> Callable[[dict], Any]:
    """Returns a callable Python function corresponding to this graph

    Args:
        dag: A DiGraph containing node names as keys, and NodeSpec objects
             in the 'node_spec' attribute

    Returns:
        The corresponding Python function for this graph
    """

    ggen = nx.topological_sort(dag)

    node_dict = {node: dag.nodes[node]["node_spec"] for node in ggen}

    def compute_from_params(**sources):
        out_p = {}

        # sources = kwargs.copy()

        sources[local_source_name] = out_p

        for node, node_spec in node_dict.items():  # ggen:
            out_p[node] = node_spec.evaluate(**sources)

        if include_inputs:
            out_p.update({k: v for k, v in sources.items() if k != "graph_locals"})

        return out_p

    if targets is None:
        return compute_from_params
    else:

        def compute_for_keys(**kwargs):
            results = compute_from_params(**kwargs)
            return {k: results[k] for k in targets}

        return compute_for_keys


class ComputeGraph:
    """A thin object oriented wrapper around the graph management functions"""

    def __init__(
        self,
        graph_dict: dict,
        local_source_name="graph_locals",
        is_traced=False,
        targets=None,
        validate_keys=True,
    ):
        """Build a fully traced DiGraph from the supplied dict

        Args:
            graph_dict: A dict with keys as node names, and arguments of
        """
        if isinstance(graph_dict, GraphObject):
            self._targets = ["out"]
            graph_dict, _ = trace_with_named_keys({"out": graph_dict}, validate_keys)
        elif not is_traced:
            self._targets = list(graph_dict.keys())
            graph_dict, _ = trace_with_named_keys(graph_dict, validate_keys)
        else:
            self._targets = []

        if targets is not None:
            self._targets = targets

        self.dag = build_dag(graph_dict, local_source_name)
        self.local_source_name = local_source_name
        self.dict = graph_dict

    def draw(self, targets=None, **kwargs):
        if targets is None:
            targets = self._targets
        return draw_compute_graph(self.dag, targets, **kwargs)

    def freeze(
        self,
        dynamic_inputs: List[Variable],
        input_variables: dict = None,
        targets: List[str] = None,
    ):
        from .dynamic import freeze_graph

        if targets is None:
            targets = self._targets

        return freeze_graph(self, targets, dynamic_inputs, input_variables)

    def filter(self, targets=None, sources=None, exclude=None):
        if targets is None and sources is None:
            targets = self._targets
        return filter_graph(self, targets, sources, exclude)

    def query(self, pattern: str) -> List[str]:
        """Return a list of keys from this graph which match the supplied regex pattern

        Args:
            pattern: Regex pattern

        Returns:
            List[str]: List of matching keys
        """
        return query(self, pattern)

    def get_input_variables(self):
        return get_input_variables(self.dag)

    def get_callable(
        self, targets=None, include_inputs=False, output_all=False
    ) -> Callable[[dict], Any]:
        if output_all:
            if targets:
                raise Exception("Target list supplied yet all outputs requested")
        else:
            if targets is None:
                targets = self._targets

        return build_callable(self.dag, targets, include_inputs, self.local_source_name)
