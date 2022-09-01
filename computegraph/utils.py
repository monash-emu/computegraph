"""
Utility functions
"""

from typing import List, Tuple, Any, Callable
import networkx as nx

from .types import Variable, Function, Data, NodeSpec, GraphObject, local

from re import match


def defer(func: callable) -> Callable:
    """Simple wrapper to defer a function call as a Function object instead
    e.g.
    defer(lambda x,y: x**y)(5.1, param("y"))
    returns a Function that lazily evaluates on param("y")

    Args:
        func: Callable to defer
    """

    def wrapped_maker(*args, **kwargs):
        return Function(func, args, kwargs)

    return wrapped_maker


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
            return [obj]
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
    if is_var(arg):
        out_key = nested_key(arg.key, layer)
        if not is_var(arg, "graph_locals"):
            if not nest_inputs:
                out_key = arg.key
            if arg.source in param_map:
                if arg.key in param_map[arg.source]:
                    out_key = param_map[arg.source][arg.key]
        return Variable(out_key, arg.source)
    else:
        return arg


def relabel_arg(arg, source, new_source):
    if is_var(arg, source):
        return Variable(arg.key, new_source)
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
        new_args.append(get_nested_arg(arg, layer, nest_inputs, param_map))
    for k, arg in f.kwargs.items():
        new_kwargs[k] = get_nested_arg(arg, layer, nest_inputs, param_map)
    return Function(f.func, new_args, new_kwargs)


def get_relabelled_func(f: Function, source: str, new_source: str) -> Function:
    """Return a Function where variables with source: source are relabelled as new_source

    Args:
        f (Function): The Function to relabel
        source: Variable source to relabel
        new_source: New label

    Returns:
        Function: The relabelled Function
    """

    new_args = []
    new_kwargs = {}
    for arg in f.args:
        new_args.append(relabel_arg(arg, source, new_source))
    for k, arg in f.kwargs.items():
        new_kwargs[k] = relabel_arg(arg, source, new_source)
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


def _get_name(obj, mapped_names):
    if obj not in mapped_names:
        var_name = obj.node_name or f"_var{len(mapped_names)}"
        if var_name in mapped_names.values():
            existing = invert_dict(mapped_names)[var_name]
            msg = f"Object with {var_name} already exists in mapping"
            raise KeyError(msg, obj, existing)
        mapped_names[obj] = var_name
    else:
        var_name = mapped_names[obj]
    return var_name


def trace_func(f, arg_table, mapped_names):
    out_args = []
    out_kwargs = {}
    for arg in f.args:
        if isinstance(arg, GraphObject):
            if is_var(arg, "graph_locals"):
                out_args.append(arg)
            else:
                var_name = _get_name(arg, mapped_names)
                arg_table[var_name] = arg
                out_args.append(local(var_name))
                trace_object(arg, arg_table, mapped_names)
        else:
            out_args.append(arg)
    for k, arg in f.kwargs.items():
        if isinstance(arg, GraphObject):
            if is_var(arg, "graph_locals"):
                out_kwargs[k] = arg
            else:
                var_name = _get_name(arg, mapped_names)
                arg_table[var_name] = arg
                out_kwargs[k] = local(var_name)
                trace_object(arg, arg_table, mapped_names)
        else:
            out_kwargs[k] = arg

    var_name = _get_name(f, mapped_names)
    arg_table[var_name] = Function(f.func, tuple(out_args), out_kwargs)
    arg_table[var_name].node_name = f.node_name


def trace_object(obj, arg_table=None, mapped_names=None):
    arg_table = arg_table or {}
    mapped_names = mapped_names or {}
    if isinstance(obj, Function):
        trace_func(obj, arg_table, mapped_names)
    elif isinstance(obj, Variable):
        var_name = _get_name(obj, mapped_names)
        arg_table[var_name] = obj
    elif isinstance(obj, Data):
        var_name = _get_name(obj, mapped_names)
        arg_table[var_name] = obj
    return arg_table, mapped_names


def invert_dict(d):
    return {v: k for k, v in d.items()}


def assign(x):
    return x


def trace_with_named_keys(in_graph, validate_keys=True):
    g = {}
    m = {}  # invert_dict(in_graph)
    for k, v in in_graph.items():
        g, m = trace_object(v, g, m)

    if validate_keys:
        for k in in_graph:
            if k in g:
                msg = f"Object with out key {k} already in graph"
                raise KeyError(msg)

    for k, v in in_graph.items():
        g[k] = Function(assign, [local(m[v])])
    return g, m


def filter_graph(cg, targets=None, sources=None, exclude=None):
    """Return a ComputeGraph that contains all targets and all sources,
    and all their interdependencies, but no extraneous nodes.
    The graph will be computable - ie ancestors of sources (and their children)
    will be included
    Any nodes dependant on exclude will be removed

    Args:
        cg: ComputeGraph to filter
        targets: Set or object convertable to set
        sources: Set or object convertable to set
        exclude: Set or object convertable to set

    Raises:
        Exception: Requires at least one argument

    Returns:
        The filtered ComputeGraph
    """
    if not targets and not sources and not exclude:
        raise Exception("At least one argument must be supplied")

    if not exclude:
        excluded = set()
    else:
        if isinstance(exclude, str):
            exclude = set((exclude,))
        exclude = set(exclude)
        excluded = exclude.copy()
        for n in exclude:
            excluded = excluded.union(nx.descendants(cg.dag, n))

    # We only have an exclude - return everything _except_ this
    if not targets and not sources:
        nodes = set(cg.dag)
    else:
        if targets is None:
            targets = set()
        if sources is None:
            sources = set()
        if isinstance(targets, str):
            targets = set((targets,))
        if isinstance(sources, str):
            sources = set((sources,))

        targets = set(targets)
        sources = set(sources)

        nodes = targets.union(sources)

        for s in sources:
            nodes = nodes.union(nx.descendants(cg.dag, s))
        for n in list(nodes):
            nodes = nodes.union(nx.ancestors(cg.dag, n))

        for t in targets:
            nodes = nodes.union(nx.ancestors(cg.dag, t))

    nodes = nodes.difference(excluded)

    out_dict = {k: v for k, v in cg.dict.items() if k in nodes}

    from .graph import ComputeGraph

    final_targets = targets.difference(excluded)

    return ComputeGraph(out_dict, is_traced=True, targets=final_targets)


def query(cg, pattern):
    return [k for k in cg.dag if match(pattern, k)]
