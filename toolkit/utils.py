import torch
import logging
import time

def normalise_or(r):
    norm = torch.linalg.norm(r)

    if norm == 0:
        return torch.zeros(len(r))
    return r / norm

def normalise(embedding):
           
        return torch.stack(
            list(map(
                normalise_or,
                embedding
            ))
        )

def get_operator(matrix):

    def operator(embedding):

        return torch.matmul(
            matrix,
            embedding
        )

    return operator

def combine_operators(op2, op1):
    
    def combined(embedding):
        return op2(op1(embedding))

    return combined

def hyperedges_to_edges(hyperedges):
    edges = []

    for hyperedge in hyperedges:
        edges += [(x,y) for x in hyperedge for y in hyperedge if x < y]

    return list(set(edges))

def mat_density(mat):
    nonzero = len(mat.values())
    elements = mat.size()[0] * mat.size()[1]

    return nonzero / elements

def mm_csr_csc(mat1, mat2):
    if mat1.layout != torch.sparse_csr or mat2.layout != torch.sparse_csc:
        raise Exception('ensure that mat1 is of layout csr and that \
        mat2 is of layout csc')

    if mat1.size()[1] != mat2.size()[0]:
        raise Exception('ensure that mat1 and mat2 have compatible dimesnions.')

    (inter, col_num) = mat2.size()

    col_num = mat2.size()[1]
    cols = []

    logging.info(f' Creating dense matrix of size: rows: {mat1.size(0)}, cols: {mat2.size()[1]}.')

    ccol_indices = mat2.ccol_indices()
    row_indices = mat2.row_indices()
    values = mat2.values()

    for i in range(col_num):
        extracted_col = torch.zeros((inter, 1), dtype=torch.float32)

        ix_range = range(ccol_indices[i], ccol_indices[i+1])

        for j in ix_range:
            extracted_col[row_indices[j]] = values[j]

        cols.append(mat1.float().matmul(extracted_col))

    return torch.cat(cols, 1)

def mm_csc_csr(mat1, mat2):
    if mat1.layout != torch.sparse_csc or mat2.layout != torch.sparse_csr:
        raise Exception('ensure that mat1 is of layout csc and that \
        mat2 is of layout csr')

    if mat1.size()[1] != mat2.size()[0]:
        raise Exception('ensure that mat1 and mat2 have compatible dimesnions.')


    # just for the sake of having a working demo, I just use the other function,
    # but this conversion probably takes at least linear time.

    start = time.time()

    result = mm_csr_csc(
        mat1.to_sparse_coo().to_sparse_csr(),
        mat2.to_sparse_coo().to_sparse_csc()
    )

    logging.info(f' It took {time.time() - start} to compute this product (1).')

    return result

def mm_csc_csr2(mat1, mat2):
    if mat1.layout != torch.sparse_csc or mat2.layout != torch.sparse_csr:
        raise Exception('ensure that mat1 is of layout csc and that \
        mat2 is of layout csr')

    if mat1.size()[1] != mat2.size()[0]:
        raise Exception('ensure that mat1 and mat2 have compatible dimesnions.')

    start = time.time()

    (row_num, inter) = mat1.size()
    col_num = mat2.size()[1]

    result = torch.zeros((row_num, col_num))

    ccol_indices = mat1.ccol_indices()
    row_indices = mat1.row_indices()
    values1 = mat1.values()
    crow_indices = mat2.crow_indices()
    col_indices = mat2.col_indices()
    values2 = mat2.values()

    for i in range(inter):

        extraceted_col = torch.zeros((row_num, 1))
        extracted_row = torch.zeros((1, col_num))
    
        iy_range = range(ccol_indices[i], ccol_indices[i+1])
        jx_range = range(crow_indices[i], crow_indices[i+1])

        for i in iy_range:
            extraceted_col[row_indices[i]][0] = values1[i]

        for j in jx_range:
            extracted_row[0][col_indices[j]] = values2[j]


        result += torch.cat([extracted_row[0][k] * extraceted_col for k in range(col_num)])

    logging.info(f'It took {time.time() - start} to compute this product (1).')

    return result

