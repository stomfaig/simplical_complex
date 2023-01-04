import toolkit.simplex as simplex
import pandas as pd
from tqdm import tqdm
import logging

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



def experiment_driver(max_simplex_dim, treshold, embedding_depths: list):


    # import
    pass

    
def run(hyperedges, max_simplex_dim, threshold, embedding_depths: list):

    try:

        s = simplex.SimplicalComplex(hyperedges, max_simplex_dim, threshold)

        for depth in embedding_depths

    except:
        logging.warn(f' An error occured while running: ')
