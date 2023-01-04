def hypergraph_link_prediction(vertex_count, hypergraph, train_ratio=0.8, test_edges_num=100, test_vertices_num=1000):
    
    training_edges_flag = False
    removed = []
    fake_edges_flag = False
    fake_edges = []

    while not (training_edges_flag and fake_edges_flag):
        x = random.randint(0, vertex_count)
        y = random.randint(0, vertex_count)

        print(f'fake edges: {len(fake_edges)}, need: {0.05 * train_ratio * (vertex_count ** 2)} | \
            removed: {len(removed)}, need: {0.05 * (1 - train_ratio) * (vertex_count ** 2)}')

        start_index = -1
        for (i, hyperedge) in enumerate(hypergraph):
            if (x in hyperedge) and (y in hyperedge):
                start_index = i
                break
        else:
            if len(fake_edges) < 0.05 * train_ratio * (vertex_count ** 2):
                fake_edges.append((x, y))
            else:
                fake_edges_flag = True
            continue
        
        if len(removed) < 0.05 * (1 - train_ratio) * (vertex_count ** 2):
            removed.append((x, y))
            for hyperedge in hypergraph[start_index:]:
                if (x in hyperedge) and (y in hyperedge):
                    hypergraph.remove(hyperedge)
                    hypergraph.extend([
                        hyperedge.remove(x),
                        hyperedge.remove(y)
                    ])
        else:
            training_edges_flag = True

    train_edges = hyperedges_to_edges(hypergraph)
    g = ig.Graph(train_edges)
    test_edges = random.sample(
        removed,
        test_edges_num
    )
    vs = list(
        map(
            lambda v: v.index,
            sorted(
                g.vs, 
                key=lambda v: v.degree()
            )
        )
    )

    link_prediction = link_prediction_curry(g, train_edges, test_edges, fake_edges, vs, test_vertices_num)
    

    return (
        g,
        hypergraph,
        link_prediction
    )