"""
Utility functions
"""

from typing import List, Tuple
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
        if source:

            def check_var(v):
                return isinstance(v, Variable) and (v.source == source)

        else:

            def check_var(v):
                return isinstance(v, Variable)

        vars = [a.name for a in obj.args if check_var(a)]
        vars += [v.name for v in obj.kwargs.values() if check_var(v)]
        return vars
    else:
        return []
