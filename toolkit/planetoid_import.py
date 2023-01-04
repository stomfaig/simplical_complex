import igraph as ig
import torch
import sys
import igraph as ig
import torch
import os
import random
from torch_geometric.datasets import Planetoid


def planetoid_import(dataset_name):

    g = ig.Graph()

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
    dataset = Planetoid(path, dataset_name, "public")
    data = dataset[0]

    v_num = len(data.x[:-1])
    print(f"Number of vertices: {v_num}")
    
    edge_indexes = data.edge_index
    sources = edge_indexes[0]
    targets = edge_indexes[1]

    print(f"Graph is directed: {data.is_directed()}")
    if data.is_directed():
        print(f"Number of edges: {len(sources) / 2}")
    else:
        print(f"Number of edges: {len(sources)}")
    edges = set()
    if data.is_undirected():
        for idx in range(len(sources)):
            if sources[idx] < targets[idx]:
                edges.add((sources[idx].item(), targets[idx].item()))
    else:
        for idx in range(len(sources)):
            edges.add((sources[idx].item(), targets[idx].item()))


    """
    This part removes the empty vertices and scales the graph down. We have to do this, even if there are no isolated vertices in the graph,
    since the vertex labeling might not be consistent, i.e. although there are v vertices, the max vertex id can be v<.
    """

    scale_graph = ig.Graph()
    scale_graph.add_vertices(v_num + 1)
    scale_graph.add_edges(list(edges))
    scale_graph.delete_vertices(scale_graph.vs.select(_degree=0))

    vertex_count = len(scale_graph.vs)
    edges_list_t = list(
        map(
            lambda e: (e.source, e.target),
            iter(scale_graph.es)
        )
    )

    return vertex_count, edges_list_t