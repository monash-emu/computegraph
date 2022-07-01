from typing import Callable, Any, Optional, List

import networkx as nx

from .types import Variable, Function, GraphDict

"""
Graph building functions
"""


def build_dag(graph_dict: GraphDict, local_source_name="graph_locals") -> nx.DiGraph:
    """Build a DiGraph from the supplied GraphDict

    Args:
        graph_dict (GraphDict): _description_

    Returns:
        nx.DiGraph: A fully traced computegraph DiGraph
    """
    dag = nx.digraph.DiGraph()
    for name, node_spec in graph_dict.items():
        dag.add_node(name, node_spec=node_spec)
    trace_edges(dag, local_source_name)
    return dag


def trace_edges(dag: nx.DiGraph, local_source_name="graph_locals"):
    """Trace the DAG and create the appropriate edges (in-place)

    Args:
        dag: A DiGraph whose keys are names, with attribute 'node_spec'
    """

    def find_source_node(dag: nx.DiGraph, var: Variable):
        source_key = var.name
        source_node = None
        if source_key in dag:
            source_node = source_key
        else:
            split_k = source_key.split(".")
            for i in range(1, len(split_k)):
                # Find at a higher layer that does exist
                layer_k = ".".join(split_k[:-i])
                if layer_k in dag:
                    source_node = layer_k
                    break
        if source_node is None:
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
                if isinstance(a, Variable) and (a.source == local_source_name):
                    # We have a nested key
                    source_node = find_source_node(dag, v)
                    dag.add_edge(source_node, node)


"""
Operations over built DiGraphs
"""


def build_callable(dag: nx.DiGraph, local_source_name="graph_locals") -> Callable[[dict], Any]:
    """Returns a callable Python function corresponding to this graph

    Args:
        dag: A DiGraph containing node names as keys, and NodeSpec objects
             in the 'node_spec' attribute

    Returns:
        The corresponding Python function for this graph
    """

    def compute_from_params(**kwargs):
        out_p = {}
        ggen = nx.topological_sort(dag)
        sources = kwargs
        for node in ggen:
            node_spec = dag.nodes[node]["node_spec"]
            if isinstance(node_spec, Variable):
                out_p[node] = kwargs[node_spec.source][node_spec.name]
            elif isinstance(node_spec, Function):
                sources.update({local_source_name: out_p})
                out_p[node] = node_spec.call(sources)
            else:
                raise Exception("Unsupported node type", node, node_spec, type(node_spec))
        return out_p

    return compute_from_params


def draw_graph(dag: nx.DiGraph, targets: Optional[List[str]] = None):
    """A basic plotting function for viewing the data flow in a graph

    Args:
        dag: A computegraph DiGraph
        targets: A list of output nodes to mark with a separate colour
    """
    pos = nx.nx_agraph.graphviz_layout(dag, prog="dot")
    node_specs = nx.get_node_attributes(dag, "node_spec")

    labels = {k: k for k in dag}

    if targets is None:
        targets = []

    def get_color(name, param):
        if isinstance(param, Variable):
            return "lightgreen"
        elif name in targets:
            return "#ee88ee"
        else:
            return "lightblue"

    node_colors = [get_color(name, param) for name, param in node_specs.items()]
    return nx.draw(dag, pos=pos, labels=labels, node_color=node_colors, width=0.4, node_size=500)


"""
Object Oriented Wrappers
"""


class ComputeGraph:
    """A thin object oriented wrapper around the graph management functions"""

    def __init__(self, graph_dict: dict, local_source_name="graph_locals"):
        """Build a fully traced DiGraph from the supplied dict

        Args:
            graph_dict: A dict with keys as node names, and arguments of
        """
        self.dag = build_dag(graph_dict, local_source_name)
        self.local_source_name = local_source_name

    def draw(self, targets=None):
        return draw_graph(self.dag, targets)

    def get_callable(self) -> Callable[[dict], Any]:
        return build_callable(self.dag, self.local_source_name)
