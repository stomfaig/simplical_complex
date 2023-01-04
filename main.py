import pandas as pd
from tqdm import tqdm
import logging

from toolkit import simplex, link_prediction
from toolkit.utils import hyperedges_to_edges



def read_zabka_csv(csv_path):
    df = pd.read_csv(csv_path)
    nodes = []
    hyper_edges = []
    edges = []
    logging.info(f" Reading baskets from zabka dataset")
    for order_id in tqdm(df["basket"].drop_duplicates()):
        basket = df[df["basket"] == order_id]["product_id"].values.tolist()
        hyper_edges.append(basket)
        nodes += basket
        edges += [(x, y) for x in basket for y in basket if x < y]
    nodes = list(set(nodes))
    edges = list(set(edges))
    return len(nodes), edges, hyper_edges

def exp1():
    data = experiment_driver(
        [3 + i for i in range(10)],
        [50 + 5 * i for i in range(10)],
        [3 + i for i in range(3)]
    )

def experiment_driver(max_simplex_dims: list, tresholds: list, embedding_depths: list):
    (nodes_num, edges, hyperedges) = read_zabka_csv('~/Downloads/zabka.csv')

    for max_simplex_dim in max_simplex_dims:
        for treshold in tresholds:
            run(hyperedges, max_simplex_dim, treshold, embedding_depths)

    
def run(hyperedges, max_simplex_dim, threshold, embedding_depths: list):

        s = simplex.SimplicalComplex(hyperedges, max_simplex_dim, threshold)

        edges = hyperedges_to_edges(s.hypergraph)

        edges_remapped = [
            tuple([s.node_persistor[simplex.Label([v])] for v in edge])
                for edge in edges
        ]

        logging.info(f' Number of edges left in the graph after truncating: {len(edges_remapped)}')

        (g, link_prediction_metric) = link_prediction.link_prediction_setup(
            len(s.node_persistor),
            edges_remapped
        )


        for depth in embedding_depths:
            embedding = simplex.generate_embedding(s, depth)

            score = link_prediction_metric(embedding)

            logging.info(f' Score: {score}')


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    exp1()