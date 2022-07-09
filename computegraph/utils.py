"""
Utility functions
"""

from typing import List, Tuple, Any
import networkx as nx

from .types import Variable, Function, NodeSpec


def build_args(args: tuple, kwargs: dict, sources: dict) -> Tuple[Tuple, dict]:
    """Return a realised args,kwargs pair containing
    actual values used a computation, based on their NodeSpec descriptions

    Args:
        args: Args tuple containing either Variables or Python data
        kwargs: Kwargs dict containing either Variables or Python data
        sources: Dictionary of dictionaries containing the lookup values for Variables

    Returns:
        Realised (args, kwargs) tuple for use in a function call
    """
    out_args = []
    for a in args:
        if isinstance(a, Variable):
            out_args.append(sources[a.source][a.name])
        else:
            out_args.append(a)
    out_kwargs = {}
    for k, v in kwargs.items():
        if isinstance(v, Variable):
            out_kwargs[k] = sources[v.source][v.name]
        else:
            out_kwargs[k] = v
    return out_args, out_kwargs


def extract_variables(
    obj: NodeSpec, source: str = None, exclude: str = "graph_locals"
) -> List[str]:
    """Return the names (keys) for all Variables referenced by a NodeSpec

    Args:
        obj: NodeSpec object (Variable or Function)
        source: Filter to apply such that only Variables from this source are returned

    Returns:
        List of keys
    """

    assert source != exclude

    def should_exclude(arg):
        if exclude:
            return is_var(arg, exclude)
        else:
            return False

    if isinstance(obj, Variable):
        if source:
            if obj.source == source:
                return [obj]
            else:
                return []
        else:
            return [obj.name]
    elif isinstance(obj, Function):
        vars = [a for a in obj.args if is_var(a, source) and not should_exclude(a)]
        vars += [v for v in obj.kwargs.values() if is_var(v, source) and not should_exclude(v)]
        return vars
    else:
        return []


def is_var(obj: Any, source: str = None) -> bool:
    """Return True is obj is a Variable that (if supplied)
       matches source

    Args:
        obj: The object of iterest
        source (str, optional): Optional source to match

    Returns:
        bool: The match status
    """
    if source:
        return isinstance(obj, Variable) and (obj.source == source)
    else:
        return isinstance(obj, Variable)


def nested_key(k, layer):
    return f"{layer}.{k}"


def get_nested_arg(arg, layer, nest_inputs, param_map):
    # if is_var(arg, "graph_locals"):
    #    return local(nested_key(arg.name, layer))
    # elif is_var(arg, "parameters")
    if is_var(arg):
        out_key = nested_key(arg.name, layer)
        if not is_var(arg, "graph_locals"):
            if not nest_inputs:
                out_key = arg.name
            if arg.source in param_map:
                if arg.name in param_map[arg.source]:
                    out_key = param_map[arg.source][arg.name]
        return Variable(out_key, arg.source)
    else:
        return arg


def get_nested_func(f: Function, layer: str, nest_inputs: bool = False, param_map=None) -> Function:
    """
    Return a Function whose graph_local Variables are renamed to be nested within layer
    """
    param_map = param_map or {}

    new_args = []
    new_kwargs = {}
    for arg in f.args:
        # if is_var(a, "graph_locals"):
        #    new_args.append(local(nested_key(a.name, layer)))
        # else:
        #    new_args.append(a)
        new_args.append(get_nested_arg(arg, layer, nest_inputs, param_map))
    for k, arg in f.kwargs.items():
        # if is_var(v, "graph_locals"):
        #    new_kwargs[k] = local(nested_key(v.name, layer))
        # else:
        #    new_kwargs[k] = v
        new_kwargs[k] = get_nested_arg(arg, layer, nest_inputs, param_map)
    return Function(f.func, new_args, new_kwargs)


def get_nested_graph_dict(
    pdict: dict, layer: str, nest_inputs: bool = False, param_map: dict = None
) -> dict:
    """Return a new computegraph dictionary (not DAG), whose keys
       are nested as <layer>.<original_key>, and whose parameters
       are also nested, unless mapped by param_map

    Args:
        pdict: The source dictionary
        layer (str): The new nesting layer
        param_map (dict, optional): Map of existing param_key:new_param_key

    Returns:
        The resulting nested dictionary
    """
    new_dict = {}

    if param_map is None:
        param_map = {}

    for k, v in pdict.items():
        k_nest = nested_key(k, layer)
        if isinstance(v, Function):
            new_dict[k_nest] = get_nested_func(v, layer, nest_inputs, param_map)
        elif is_var(v):
            new_dict[k_nest] = get_nested_arg(v, layer, nest_inputs, param_map)
        #    if k in param_map:
        #        new_key = param_map[k]
        #        new_dict[k_nest] = param(new_key)
        #    else:
        #        new_dict[k_nest] = param(k_nest)
    return new_dict


def expand_nested_dict(src: dict, layer: str = None, include_parents=False) -> dict:
    """Recursively expand a nested dictionary such that
       {'a': {'aa': 0}, 'b': 1} -> {'a.aa': 0, 'b': 1}

    Args:
        src: The nested dictionary to expand
        layer: The current nested layer
        include_parents: Include all unexpanded subdictionaries

    Returns:
        The expanded dictionary
    """

    out_dict = {}
    for k, v in src.items():
        if layer:
            out_key = nested_key(k, layer)
        else:
            out_key = k
        if isinstance(v, dict):
            if include_parents:
                out_dict[out_key] = v
            out_dict.update(expand_nested_dict(v, out_key, include_parents))
        else:
            out_dict[out_key] = v
    return out_dict


def get_input_variables(dag: nx.DiGraph):
    found_params = []
    node_specs = nx.get_node_attributes(dag, "node_spec")
    for v in node_specs.values():
        param_for_node = extract_variables(v)
        found_params += param_for_node

    return set(found_params)


def get_with_injected_parameters(dag):
    out_dag = dag.copy()
    found_params = []
    pmap = {}
    node_specs = nx.get_node_attributes(dag, "node_spec")
    for k, v in node_specs.items():
        param_for_node = extract_variables(v)
        pmap[k] = param_for_node
        found_params += param_for_node

    found_params = set(found_params)
    for p in found_params:
        out_dag.add_node(f"{p.source}.{p.name}", node_spec=p)
    for k, v in pmap.items():
        for p in v:
            out_dag.add_edge(f"{p.source}.{p.name}", k)
    return out_dag
