"""
Basic data structures used throughout the rest of the codebase
There are 2 primary types - Variable and Function
"""

from typing import Tuple, Union, Dict
from abc import ABC, abstractmethod
from collections.abc import Hashable

from reprlib import repr as rrepr

import numpy as np

# from jax import numpy as fnp
fnp = np


class AbstractGraphObject(ABC):
    node_name = None

    @abstractmethod
    def evaluate(self, **sources):
        pass


class GraphObject(AbstractGraphObject):
    def __add__(self, other):
        return Function(fnp.add, [self, other])

    def __radd__(self, other):
        return Function(fnp.add, [other, self])

    def __mul__(self, other):
        return Function(fnp.multiply, [self, other])

    def __rmul__(self, other):
        return Function(fnp.multiply, [other, self])

    def __sub__(self, other):
        return Function(fnp.subtract, [self, other])

    def __rsub__(self, other):
        return Function(fnp.subtract, [other, self])

    def __truediv__(self, other):
        return Function(fnp.divide, (self, other))

    def __rtruediv__(self, other):
        return Function(fnp.divide, (other, self))

    def __pow__(self, other):
        return Function(fnp.power, (self, other))

    def __rpow__(self, other):
        return Function(fnp.power, (self.other))

    def __getattr__(self, attr):
        try:
            np_attr = getattr(np, attr)
        except AttributeError:
            raise AttributeError(f"No attribute {attr} available", self)
        if isinstance(np_attr, np.ufunc):
            fnp_attr = getattr(fnp, attr)
            return lambda: Function(fnp_attr, [self])
        else:
            raise AttributeError(f"No attribute {attr} available", self)


class Variable(GraphObject):
    """The basic type for all variables used in graphs
    They are identifiable via their name (unique to a given source),
    and their source, which is a user definable lookup for dictionaries
    of arguments passed to functions.
    """

    def __init__(self, key: str, source: str):
        self.key = key
        self.source = source
        self.node_name = f"{source}.{key}"

    def __hash__(self):
        return hash((self.key, self.source))

    def __repr__(self):
        return f"Variable: {self.source}[{self.key}]"

    def __eq__(self, other):
        if isinstance(other, Variable):
            return (self.key == other.key) and (self.source == other.source)
        else:
            return False

    def evaluate(self, **sources):
        return sources[self.source][self.key]


class Data(GraphObject):
    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return f"Data: {rrepr(self.data)}"

    def evaluate(self, **sources):
        return self.data

    def __hash__(self):
        if isinstance(self.data, Hashable):
            return hash(self.data)
        else:
            return id(self.data)

    def __eq__(self, other):
        equality = self.data == other.data
        if isinstance(equality, bool):
            return equality
        else:
            return all(equality)


def local(key: str) -> Variable:
    """Convenience function for returning a graph_locals variable

    Args:
        key: Variable Name

    Returns:
        Variable with source "graph_locals" (default graph locals lookup dict)
    """
    return Variable(key, source="graph_locals")


def param(key: str) -> Variable:
    """Convenience function for returning a parameters variable

    Args:
        key: Variable Name

    Returns:
        Variable with source "parameters"
    """
    return Variable(key, source="parameters")


class Function(GraphObject):
    """Universal wrapper for callable functions, whose args and kwargs
    may be either GraphObject (resolved at run-time), or any other Python value
    (regarded as constant)
    """

    def __init__(self, func: callable, args: tuple = None, kwargs: dict = None):
        self.func = func
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        if not (isinstance(args, tuple) or isinstance(args, list)):
            raise TypeError("Args must be list or tuple", args)
        self.args = tuple(args)
        self.kwargs = kwargs

    def __hash__(self):
        return hash((self.func, self.args, tuple(self.kwargs.items())))

    def __repr__(self):
        return f"Function: {rrepr(self.func.__name__)}, args={rrepr(self.args)}, kwargs={rrepr(self.kwargs)})"

    def __eq__(self, other):
        if isinstance(other, Function):
            return (
                self.func == other.func and self.args == other.args and self.kwargs == other.kwargs
            )
        else:
            return False

    def build_args(self, sources: dict) -> Tuple[tuple, dict]:
        return build_args(self.args, self.kwargs, sources)

    def evaluate(self, **sources):
        args, kwargs = self.build_args(sources)
        return self.func(*args, **kwargs)


def evaluate_lazy(obj, sources):
    if isinstance(obj, GraphObject):
        return obj.evaluate(**sources)
    else:
        return obj


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
        if isinstance(a, GraphObject):
            out_args.append(evaluate_lazy(a, sources))
        else:
            out_args.append(a)
    out_kwargs = {}
    for k, v in kwargs.items():
        if isinstance(v, GraphObject):
            out_kwargs[k] = evaluate_lazy(v, sources)
        else:
            out_kwargs[k] = v
    return out_args, out_kwargs


# Declare some derived types for annotation purposes
NodeSpec = Union[Variable, Function]
GraphDict = Dict[str, NodeSpec]
