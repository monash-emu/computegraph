"""
Utility functions
"""

from typing import List, Tuple, Any
from .types import Variable, Function, NodeSpec, param, local


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


def extract_variables(obj: NodeSpec, source: str = None) -> List[str]:
    """Return the names (keys) for all Variables referenced by a NodeSpec

    Args:
        obj: NodeSpec object (Variable or Function)
        source: Filter to apply such that only Variables from this source are returned

    Returns:
        List of keys
    """
    if isinstance(obj, Variable):
        if source:
            if obj.source == source:
                return [obj.name]
        else:
            return [obj.name]
    elif isinstance(obj, Function):
        vars = [a.name for a in obj.args if is_var(a, source)]
        vars += [v.name for v in obj.kwargs.values() if is_var(v, source)]
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


def get_nested_func(f: Function, layer: str) -> Function:
    """
    Return a Function whose graph_local Variables are renamed to be nested within layer
    """
    new_args = []
    new_kwargs = {}
    for a in f.args:
        if is_var(a, "graph_locals"):
            new_args.append(local(nested_key(a.name, layer)))
        else:
            new_args.append(a)
    for k, v in f.kwargs.items():
        if is_var(v, "graph_locals"):
            new_kwargs[k] = local(nested_key(v.name, layer))
        else:
            new_kwargs[k] = v
    return Function(f.func, new_args, new_kwargs)


def get_nested_graph_dict(pdict: dict, layer: str, param_map: dict = None) -> dict:
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
            new_dict[k_nest] = get_nested_func(v, layer)
        elif is_var(v, "parameters"):
            if k in param_map:
                new_key = param_map[k]
                new_dict[k_nest] = param(new_key)
            else:
                new_dict[k_nest] = param(k_nest)
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
