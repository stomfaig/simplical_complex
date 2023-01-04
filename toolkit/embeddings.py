import torch

def get_random_embedding_c(embedding_dim_tuple):
    (vc, ed) = embedding_dim_tuple
    return get_random_embedding(vc, ed)

def get_random_embedding(vertex_count, embedding_dim):
    return (2 * torch.rand(vertex_count, embedding_dim) - 1)

def get_evec_largest_embedding(matrix, embedding_dim):
    levals, levecs = torch.lobpcg(matrix, k = embedding_dim, largest=True)
    return levecs

