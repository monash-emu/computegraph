from typing import Optional, List

import networkx as nx

from computegraph import options
from computegraph.types import Variable, Function, Data

from .ngraph import get_traces


def draw_compute_graph(dag: nx.DiGraph, targets: Optional[List[str]] = None, **kwargs):
    backend = options.drawing["backend"]

    if backend == "plotly":
        return draw_computegraph_plotly(dag, targets, **kwargs)
    if backend == "matplotlib":
        return draw_compute_graph_mpl(dag, targets, **kwargs)


def draw_compute_graph_mpl(dag: nx.DiGraph, targets: Optional[List[str]] = None, **kwargs):
    """A basic plotting function for viewing the data flow in a graph

    Args:
        dag: A computegraph DiGraph
        targets: A list of output nodes to mark with a separate colour
    """
    if options.drawing["use_graphviz"]:
        pos = nx.nx_agraph.graphviz_layout(dag, prog="dot")
    else:
        pos = nx.spring_layout(dag)
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


def draw_computegraph_plotly(
    dag: nx.DiGraph, targets: Optional[List[str]] = None, title: str = None, **kwargs
):

    if options.drawing["use_graphviz"]:
        pos = nx.nx_agraph.graphviz_layout(dag, prog="dot")
    else:
        pos = nx.spring_layout(dag)

    node_specs = nx.get_node_attributes(dag, "node_spec")

    labels = [k for k in dag]

    tab_str = "&nbsp;" * 4

    def get_node_desc(name, node_spec):
        out_text = [str(name)]
        if isinstance(node_spec, Function):
            out_text += [str(node_spec.func.__name__)]
            out_text += ["args:"]
            out_text += [f"{tab_str}{arg}" for arg in node_spec.args]
            out_text += ["kwargs:"]
            out_text += [f"{tab_str}{k}: {v}" for k, v in node_spec.kwargs.items()]
        else:
            out_text += [str(node_spec)]
        return "<br>".join(out_text)

    desc = [get_node_desc(name, node_spec) for name, node_spec in node_specs.items()]

    if targets is None:
        targets = []

    def get_color(name, node_spec):
        if isinstance(node_spec, Variable):
            return "lightgreen"
        elif isinstance(node_spec, Data):
            return "red"
        elif name in targets:
            return "#ee88ee"
        else:
            return "lightblue"

    node_colors = [get_color(name, param) for name, param in node_specs.items()]

    edge_trace, node_trace = get_traces(dag, pos)

    node_trace.marker.color = node_colors
    node_trace.hovertext = desc
    node_trace.text = labels

    if title is None:
        title = "ComputeGraph"

    fig = get_graph_figure(edge_trace, node_trace, title)
    fig.update_traces(textposition="top center")

    layout = options.drawing["plotly"]["layout"]
    fig.update_layout(layout)
    if "layout" in kwargs:
        fig.update_layout(kwargs["layout"])

    return fig


def get_graph_figure(edge_trace, node_trace, title: str = None):
    """Return a plotly Figure constructed from the traces
       returned by ngraph.get_traces

    Args:
        edge_trace (_type_): _description_
        node_trace (_type_): _description_
        title: Optional title

    Returns:
        go.Figure: The resulting Figure
    """
    from plotly import graph_objects as go

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=title,
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(visible=False, showgrid=False, zeroline=False, showticklabels=False),
        ),
    )
    return fig
