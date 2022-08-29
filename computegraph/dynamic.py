from __future__ import annotations

from typing import List, Set, TYPE_CHECKING


import networkx as nx

from computegraph.types import Variable, Function, Data, GraphObject
from computegraph.utils import is_var

from computegraph import ComputeGraph


def get_graph_args(f):
    out = []
    if isinstance(f, Function):
        for a in f.args:
            if isinstance(a, GraphObject):
                out.append(a)
        for k, a in f.kwargs.items():
            if isinstance(a, GraphObject):
                out.append(a)
    return out


def freeze_graph(cg: ComputeGraph, targets, dyn_sources, fixed_in_values: dict = None):
    """

    Args:
        cg (_type_): _description_
        targets (_type_): _description_
        dyn_sources (_type_): _description_
        fixed_in_values (optional): Dictionary of fixed values to compute as Data,
            if not supplied, then a static graph to compute these later will be returned

    Returns:
        _type_: _description_
    """

    # Accept either a list of strings, or of Variables (or a mix)
    dyn_sources = set([p.node_name if isinstance(p, Variable) else p for p in dyn_sources])

    target_anc = set()
    for t in targets:
        target_anc = target_anc.union(nx.ancestors(cg.dag, t))
    source_desc = set()
    for s in dyn_sources:
        source_desc = source_desc.union(nx.descendants(cg.dag, s))

    # Full tree of ancestors that must have values, based on targets
    full_target_tree = target_anc.union(targets)

    # All nodes that are marked dynamic based on dyn_params
    dyn_must_compute = source_desc.union(dyn_sources)

    # The final dynamic tree - all those nodes that must be computed dynamically in order to fulfill targets
    final_dyn_tree = full_target_tree.intersection(dyn_must_compute)

    static_nodes = full_target_tree.difference(final_dyn_tree)

    cgt_static = {k: v for k, v in cg.dict.items() if k in static_nodes}
    cgt_dyn = {k: v for k, v in cg.dict.items() if k in final_dyn_tree}

    req_dyn_inputs = []
    for k, v in cgt_dyn.items():
        req_dyn_inputs += [a.key for a in get_graph_args(v)]
    req_dyn_inputs

    static_targets = set()
    for k in req_dyn_inputs:
        if k not in cgt_dyn:
            static_targets.add(k)

    for k in targets:
        if k not in cgt_dyn:
            static_targets.add(k)

    static_cg = ComputeGraph(cgt_static, is_traced=True, targets=static_targets)
    mixed_d = cgt_dyn.copy()

    if fixed_in_values is not None:
        static_d = static_cg.get_callable(output_all=True)(**fixed_in_values)
        for k in static_targets:
            mixed_d[k] = Data(static_d[k])

        frozen_cg = ComputeGraph(mixed_d, is_traced=True, targets=targets)
        return frozen_cg
    else:

        for k in static_targets:
            mixed_d[k] = Variable(k, "static_inputs")

        frozen_cg = ComputeGraph(mixed_d, is_traced=True, targets=targets)
        return frozen_cg, static_cg
