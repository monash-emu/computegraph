"""
Basic data structures used throughout the rest of the codebase
There are 2 primary types - Variable and Function
"""

from typing import Tuple, Union, Dict


class Variable:
    """The basic type for all variables used in graphs
    They are identifiable via their name (unique to a given source),
    and their source, which is a user definable lookup for dictionaries
    of arguments passed to functions.
    """

    def __init__(self, name: str, source: str):
        self.name = name
        self.source = source

    def __hash__(self):
        return hash((self.name, self.source))

    def __repr__(self):
        return f"Variable {self.source}[{self.name}]"

    def __eq__(self, other):
        return (self.name == other.name) and (self.source == other.source)


class Data:
    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return f"Data: {str(self.data)}"


def local(name: str) -> Variable:
    """Convenience function for returning a graph_locals variable

    Args:
        name: Variable Name

    Returns:
        Variable with source "graph_locals" (default graph locals lookup dict)
    """
    return Variable(name, source="graph_locals")


def param(name: str) -> Variable:
    """Convenience function for returning a parameters variable

    Args:
        name: Variable Name

    Returns:
        Variable with source "parameters"
    """
    return Variable(name, source="parameters")


class Function:
    """Universal wrapper for callable functions, whose args and kwargs
    may be either Variables (resolved at run-time), or any other Python value
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
        return f"Function: func={self.func}, args={self.args}, kwargs={self.kwargs})"

    def __eq__(self, other):
        return self.func == other.func and self.args == other.args and self.kwargs == other.kwargs

    def build_args(self, sources: dict) -> Tuple[tuple, dict]:
        from .utils import build_args

        return build_args(self.args, self.kwargs, sources)

    def call(self, sources: dict):
        args, kwargs = self.build_args(sources)
        return self.func(*args, **kwargs)


# Declare some derived types for annotation purposes
NodeSpec = Union[Variable, Function]
GraphDict = Dict[str, NodeSpec]
