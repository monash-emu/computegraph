import networkx as nx


from .add_edge import addEdge


def get_traces(G, pos=None):
    import plotly.graph_objects as go

    if pos is None:
        pos = nx.layout.spring_layout(G)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        start = pos[edge[0]]
        end = pos[edge[1]]
        edge_x, edge_y = addEdge(start, end, edge_x, edge_y, 0.8, "end", 5.0, 30, 20)

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1.0, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        marker=dict(
            showscale=False,
            colorscale="YlGnBu",
            reversescale=True,
            color=[],
            size=10,
            line_width=0.5,
        ),
    )

    return edge_trace, node_trace
