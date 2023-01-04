import torch
from toolkit.utils import *
from toolkit.link_prediction import link_prediction_setup
from toolkit.embeddings import *
from toolkit.planetoid_import import planetoid_import

datasets = ['Cora', 'Pubmed', 'CiteSeer']
embedding_dim = 100


"""
data = {
    'dataset_name'
    'embedding_dim',
    'vertex_count',
    'edges'
}

"""


""" Currently supported experiment macros:

get_random_embedding(vertex_count, embedding_dim)

get_graph_data(data): returns a tuple (vertex_count, edges) for the currently evaluated graph
get_embedding_dim(data) returns a tuple with (vertex_count, embedding_dim)

get_link_prediction_metric(data): returns a tuple (g, link_prediction) where g is the 'known graph'

get_adjacency_matrix(g): returns the adjacency matrix of the given graph
get_discrete_laplacian_matrix(g): returns the discrete laplacian matrix of the given graph
get_random_walk_matrix(g): returns the random walk matrix of the given graph

propagate(M, embedding, steps, normalised=False): returns a list of (steps + 1) embeddings, that correspod to [embedding, M * embedding, M^2 * embedding, ... ]

check_one(dataset, experiment): test the embedding against a given dataset
check_random(experiment): test the embedding against a random dataset
check_all(experiment): test the embedding against all the available datasets

"""

def get_graph_data(data):
    return (data['vertex_count'], data['edges'])

def get_embedding_dim(data):
    return (data['vertex_count'] + 1, data['embedding_dim'])

def get_link_prediction_metric(data):
    return link_prediction_setup(data['vertex_count'], data['edges'])


def get_adjacency_matrix(g):
    return torch.tensor(g.get_adjacency().data).type(torch.float32)

def get_discrete_laplacian_matrix(g):
    return torch.tensor(g.laplacian()).type(torch.float32)

def get_largest_evec_init(g, dim):
    return get_evec_largest_embedding(get_discrete_laplacian_matrix(g), dim)

def get_random_walk_matrix(g):
    adjacency_matrix = get_adjacency_matrix(g)
    return normalise(adjacency_matrix)


def generic_setup():
    data = {
        'embedding_dim' : embedding_dim,
    }
    return data

def propagate(operator, embedding, steps, normalise_=False):
    result = [embedding]
    for _ in range(steps):
        embedding = operator(embedding)

        if normalise_:
            normalise(embedding)

        result.append(embedding)
    return result

def check_one(dataset, experiment):
    vertex_count, edges = planetoid_import(dataset)

    data = generic_setup()
    data.update({
        'dataset_name' : dataset,
        'vertex_count' : vertex_count,
        'edges' : edges
    })

    return experiment(data)

def check_one_twice(dataset, experiment):
    vertex_count, edges = planetoid_import(dataset)

    data = generic_setup()
    data.update({
        'dataset_name' : dataset,
        'vertex_count' : vertex_count,
        'edges' : edges
    })

    return (experiment(data), experiment(data))

def check_random(experiment):
    random = 'Pubmed' #make this actually random
    check_one(random, experiment)

def check_all(experiment):
    for dataset in datasets:
        check_one(dataset, experiment)
