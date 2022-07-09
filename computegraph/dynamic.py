from __future__ import annotations

from typing import List, Set, TYPE_CHECKING


import networkx as nx

from computegraph.types import Variable, Function, Data
from computegraph.utils import is_var

if TYPE_CHECKING:
    from computegraph import ComputeGraph


def split_static_dynamic(
    graph_dict: dict, dynamic_inputs: List[Variable], targets: List[str], **kwargs
) -> ComputeGraph:
    from computegraph import ComputeGraph

    base_graph = ComputeGraph(graph_dict)
    dyn_map = get_dynamic_mappings(base_graph, dynamic_inputs, targets)
    base_runner = base_graph.get_callable()
    base_outvals = base_runner(**kwargs)

    assignment_map = get_assignment_map(dyn_map)

    fixed_data = {}
    for arg in dyn_map["fixed_data"]:
        fixed_data[arg] = Data(base_outvals[arg])

    dyn_dict = get_frozen_graph_dict(graph_dict, dyn_map["dynamic"], assignment_map, fixed_data)

    return ComputeGraph(dyn_dict)


def get_dynamic_mappings(
    cgraph: ComputeGraph, dynamic_inputs: List[Variable], targets: List[str]
) -> dict:
    dag = cgraph.pdag
    marked_dyn = set()

    dynamic_input_nodenames = [f"{dynp.source}.{dynp.name}" for dynp in dynamic_inputs]

    for dynp in dynamic_input_nodenames:
        cur_descendants = nx.descendants(dag, dynp)
        # Check to see if we care about any of these
        if any([t in cur_descendants for t in targets]):
            marked_dyn = marked_dyn.union(cur_descendants)

    fixed_targets = set(targets)

    for node in marked_dyn:
        cur_pred = set(dag.predecessors(node))
        fixed_targets = fixed_targets.union(cur_pred)

    fixed_targets_p = fixed_targets.difference(marked_dyn.union(dynamic_input_nodenames))

    # Actual non-input graph nodes
    all_nonp_nodes = set(cgraph.dag.nodes)

    fixed_targets = all_nonp_nodes.intersection(fixed_targets_p)
    dyn_targets = all_nonp_nodes.intersection(marked_dyn)

    # Static parameters to capture
    fixed_p = fixed_targets_p.difference(fixed_targets)
    # FIXME: Doing a lot of messing around with this name splitting and joining...
    fixed_p = set([Variable(p.split(".")[1], p.split(".")[0]) for p in fixed_p])

    return dict(dynamic=dyn_targets, fixed_data=fixed_targets, fixed_inputs=fixed_p)


def get_assignment_map(dmap: dict) -> dict:
    """Return an assignment_map of the kind consumed by reassign_func_sources

    Args:
        dmap: DynamicMapping (from get_dynamic_mappings)

    Returns:
        AssignmentMap
    """

    out_map = {}
    # FIXME: "fixed_p" implies 'fixed parameter', but let's call it fixed inputs or something...

    # Remap input variables such that they now look up the appropriate fixed graph_local
    for arg in dmap["fixed_inputs"]:
        if arg.source not in out_map:
            out_map[arg.source] = {}
        out_map[arg.source][arg.name] = Variable(f"{arg.source}.{arg.name}", "graph_locals")

    return out_map


def reassign_func_sources(f: Function, assignment_map: dict) -> Function:
    """
    Return a Function whose variables are reassigned based on assignment_map
    """

    new_args = []
    new_kwargs = {}
    for arg in f.args:
        out_arg = arg
        if is_var(arg):
            if arg.source in assignment_map:
                if arg.name in assignment_map[arg.source]:
                    out_arg = assignment_map[arg.source][arg.name]
        new_args.append(out_arg)
        # new_args.append(get_nested_arg(arg, layer, nest_inputs, param_map))
    for k, arg in f.kwargs.items():
        out_arg = arg
        if is_var(arg):
            if arg.source in assignment_map:
                if arg.name in assignment_map[arg.source]:
                    out_arg = assignment_map[arg.source][arg.name]
        new_kwargs[k] = out_arg
        # new_kwargs[k] = get_nested_arg(arg, layer, nest_inputs, param_map)
    return Function(f.func, new_args, new_kwargs)


def get_frozen_graph_dict(
    orig_dict: dict, dynamic_keys: Set[str], assignment_map: dict, fixed_data: dict
) -> dict:
    """Return a graph dict containing both fixed data and the requested dynamic nodes

    Args:
        orig_dict: The original fully dynamic graph dict
        dynamic_keys: List of dynamic node keys to retain in the new dict
        assignment_map: Output of get_assignment_map
        fixed_data: Dict whose values are Data nodes

    Returns:
        Frozen (mixed static/dynamic) graph dict
    """
    out_dict = {
        k: reassign_func_sources(v, assignment_map)
        for k, v in orig_dict.items()
        if k in dynamic_keys
    }
    out_dict.update(fixed_data)
    return out_dict
