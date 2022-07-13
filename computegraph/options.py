try:
    import pygraphviz  # noqa: F401

    _use_graphviz = True
except ImportError:
    _use_graphviz = False

try:
    import plotly  # noqa: F401

    _backend = "plotly"
except ImportError:
    _backend = "matplotlib"

drawing = {
    "backend": _backend,  # May be one of 'plotly', 'matplotlib'
    "use_graphviz": _use_graphviz,  # Enable or disable use of GraphViz layouts
    "plotly": {
        "layout": {
            "width": 800,
            "height": 800,
        }
    },
}
