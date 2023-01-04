import torch
import logging
from functools import cache
from tqdm import tqdm
from toolkit.utils import mm_csr_csc, mm_csc_csr, mm_csc_csr2

"""
    Simplex framework for Cleora experiments. 

    Brief documentation:
        Simplices are represented in a way similar to as described in [ref]. When creating a simplex, 
        we label each sub simplex using the Label class, whch is based around a list with some extra fun-
        ctionality. Then, we indicate all the adjacency relations in a HashMatrixBuilder, which is a sparse
        matrix data structure, only storing lines of the form:
            (index pair), value
        We do this, because we don't know ahead of time the number of simplices of different sizes, and even
        though it could be calculated, we can do without that. Then, when every index is logged, we collapse
        the matrix builder into an actual matrix, and and a Persistor that contains the additional data needed
        to match up the simplex labels and the rows of the matrix.
        To do calculations though, we have to make sure that the matrices are aligned, which is done by built
        in functionality, which given a persistor makes sure that all entries in the provided persistor exist
        in the tensor, and that their indices are aligned.
        Finally we calculate all Laplacians and cache the results, hence minimising computational cost.

"""

class Label:

    def __init__(self, letters: list):
        self.letters = letters

    def get_letters(self):
        return self.letters
    
    def append(self, letter):
        letters_copy = self.letters.copy()
        letters_copy.append(letter)
        return Label(letters_copy)

    def sublabel(self, i):
        sublabel = self.letters[:i].copy() 
        sublabel.extend(self.letters[i+1:])
        return Label(sublabel)

    def __len__(self):
        return len(self.letters)

    def __repr__(self):
        return f'Label: {self.letters}'

    def __eq__(self, other):
        return self.letters == other.letters

    def __hash__(self):
        return hash(frozenset(self.letters))

class Persistor:

    def __init__(self):
        self.persistor = {}

    def create(self, location):
        if not location in self.persistor.keys():
            self.persistor.update({location : len(self.persistor)})

    def get_or_create(self, location):
        
        self.create(location)
        return self.persistor[location]
    
    def get(self, location):
        
        if not location in self.persistor.keys():
            raise IndexError
        
        return self.persistor[location]

    def keys(self):
        return self.persistor.keys()

    def update(self, items: dict):
        self.persistor.update(items)
    
    def values(self):
        return self.persistor.values()

    def extend(self, k):
        multipersistor = MultiPersistor(k)
        multipersistor.persistors = [self for _ in range(k)]

        return multipersistor

    def remove(self, label):
        self.persistor.pop(label, None)

    def __getitem__(self, item):
        return self.persistor[item]

    def __setitem__(self, item, val):
        self.persistor[item] = val

    def __len__(self):
        return len(self.persistor)

    def __str__(self):
        return f'{self.persistor}'

class MultiPersistor:

    def __init__(self, k=0):
        self.k = k
        self.persistors = [Persistor() for _ in range(k)]

    def get_or_create(self, location: tuple):
        index = [None for _ in range(self.k)]

        for (i, location_single) in enumerate(location):
            index[i] = self.persistors[i].get_or_create(location_single)

        return tuple(index)

    def get(self, location: tuple):
        if len(location) != self.k:
            raise IndexError('location dim and persistor dim doesn\'t match')
        
        index = tuple([None for _ in range(self.k)])

        for (i, location_single) in enumerate(location):
            index[i] = self.persistors[i].get(location_single)

        return index
    
    def get_persistor(self, i):
        return self.persistors[i]

    def add_persistor(self, persistor: Persistor):
        self.k += 1
        self.persistors.append(persistor)

    def size(self):
        size = [None for _ in range(self.k)]

        for i in range(self.k):
            size[i] = len(self.persistors[i])

        return tuple(size)

    def __str__(self):
        return '\n'.join([str(persistor) for persistor in self.persistors])

class HashMatrix:
    
    def __init__(self, tensor: torch.tensor, persistor: MultiPersistor):
        self.tensor = tensor
        self.persistor = persistor

    def __getitem__(self, location: list):
        (row, col) = self.persistor.get(location)

        return self.tensor[row, col]
    
    def align(self, other: Persistor, align_dir=1):
        if self.tensor.is_sparse:
            raise ValueError('HashMatrix tensor must be dense for alignment!')

        persistor = self.persistor.get_persistor(align_dir)

        for key in other.keys():
            if not key in persistor.keys():
                
                self.tensor = torch.cat((
                        self.tensor,
                        torch.transpose(
                            torch.zeros(1, len(self.persistor.get_persistor(1 - align_dir))),
                            0,
                            align_dir
                        )
                    ),
                    align_dir
                    )
                
                persistor.update({key : (len(persistor))})

        for key in other.keys():
            if other[key] != persistor[key]:

                order = list(range(len(persistor)))

                order[other[key]] = persistor[key]
                order[persistor[key]] = other[key]

                self.tensor = torch.index_select(self.tensor, align_dir, torch.tensor(order))

                swap_key = list(persistor.keys())[list(persistor.values()).index(other[key])]
                temp = persistor[key]
                persistor[key] = other[key]
                persistor[swap_key] = temp
                


    def get_tensor(self):
        return self.tensor

    def size(self):
        return self.tensor.size()

class HashMatrixBuilder:

    def __init__(self, persistor=None, header=''):
        self.header = header
        self.items = {}
        if persistor is None:
            self.persistor = MultiPersistor(2)
        else:
            self.persistor = persistor

    def __setitem__(self, location: tuple, item):
        index = self.persistor.get_or_create(location)
        self.items.update({index : item})

    def __getitem__(self, location: tuple):
        index = self.persistor.get(location)
        return self.items[index]

    def collapse(self):
        size = self.persistor.size()
        t = torch.zeros(size[0], size[1])

        for item in self.items:
            t[int(item[0]), int(item[1])] = self.items[item]

        hashMatrix = HashMatrix(
            t,
            self.persistor
        )

        return hashMatrix

    def collapse_to_csr(self):

        if len(self.items) == 0:
            logging.warning(f' An empty incidence matrix was created with header: {self.header}') 
            return None

        row_num = len(self.persistor.get_persistor(0))

        row_counts = [0 for i in range(row_num)] 
        columns = [[] for i in range(row_num)] 
        column_values = [[] for i in range(row_num)]

        for item in self.items:
            ix = item[0]
            iy = item[1]

            row_counts[ix] += 1
            columns[ix].append(iy)
            column_values[ix].append(self.items[item])


        crow_indices = torch.tensor([sum(row_counts[0:i]) for i in range(len(row_counts)+1)], dtype=torch.long)
        col_indices = torch.cat([torch.tensor(l) for l in columns]).type(torch.long)
        values = torch.cat([torch.tensor(l) for l in column_values])

        if len(values) > 0:

            sparse_csr = torch.sparse_csr_tensor(
                    crow_indices,
                    col_indices,
                    values,
                    dtype=torch.float64
                )

            return HashMatrix(
                sparse_csr,
                self.persistor
            )      

        return None

class SimplexNode:

    def __init__(self, parent, label: Label, id):
        self.parent = parent
        self.children = {}
        self.label = label
        self.depth = len(self.label) - 1
        self.id = id
        self.weight = 0

    def insert_simplex(self, weight, prefix: Label, hyperedge: list, get_next_id):
        self.weight += weight
        
        if len(hyperedge) == 0:
            return
        
        root = hyperedge[0]
        label = prefix.append(root)
        if not root in self.children.keys():
            node = SimplexNode(self, label, get_next_id())
            self.children.update({root : node})
        
        if len(hyperedge) == 1:
            return 

        self.children[root].insert_simplex(weight, label, hyperedge[1:], get_next_id)
        self.insert_simplex(weight, prefix, hyperedge[1:], get_next_id)

    def threshold_filter(self, threshold):
        # see https://stackoverflow.com/questions/11941817/how-to-avoid-runtimeerror-dictionary-changed-size-during-iteration-error
        for child in list(self.children):
            if self.children[child].weight < threshold:
                if self.children[child].depth > 0:
                    self.children.pop(child)
                    continue
            self.children[child].threshold_filter(threshold)
            

    def log_incidence(self, incidence_matrices):
        
        if self.depth > 0:
            for i in range(len(self.label)):
                incidence_matrices[self.depth-1][self.label, self.label.sublabel(i)] = 1
        
        for child in self.children:
            self.children[child].log_incidence(incidence_matrices)
        

    def __repr__(self):
        return f'children of [{self.label}] : {self.children}'

class SimplicalComplex(SimplexNode):
    
    """
        TODO: work on naming conventions here
              speed up simplex generation by compacting the recursive part
            
    """

    def __init__(self, hypergraph: list, max_simplex_dim=5, threshold = 50):
        self.next_id = 0
        self.label= Label([])
        self.depth = -1
        self.children = {}
        self.weight = 0

        hypergraph = list(filter(lambda x: len(x) <= max_simplex_dim, hypergraph))
        hypergraph.sort(key=len)
        self.hypergraph = hypergraph
        self.n = min(max_simplex_dim, len(hypergraph[-1]))

        logging.info(f' Creating Simplical Complex with \n \
            \t max_simplex_dim : {max_simplex_dim} \n \
            \t threshold : {threshold} \
            ')
        for hyperedge in tqdm(hypergraph):
            hyperedge.sort()
            empty_label = Label([])
            self.insert_simplex(1/len(hyperedge), empty_label, hyperedge, self.get_next_id)

        self.threshold_filter(threshold)

        incidence_matrices = get_cross_persisted_matrices(self.n - 1)
        self.log_incidence(incidence_matrices)
        incidence_matrices = [l.collapse_to_csr() for l in incidence_matrices]
        incidence_matrices = [l for l in incidence_matrices if l != None]
        self.n = len(incidence_matrices)
        self.incidence_matrices = incidence_matrices

        # have to create a big persistor for all the depth 0 simplices, ie nodes
        self.node_persistor = Persistor()
        for child in self.children:
            self.node_persistor.create(
                self.children[child].label
            )


    @cache
    def get_k_laplacian(self, k):

        T = self.incidence_matrices[k].get_tensor()
        L = mm_csr_csc(
                T,
                torch.t(T)
            )

        if k == 0:
            return HashMatrix(L, self.incidence_matrices[k].persistor.get_persistor(0).extend(2))

        t = self.incidence_matrices[k-1].get_tensor()
        l = mm_csc_csr(
                torch.t(t),
                t
            )

        if k == self.n:
            return HashMatrix(l, self.incidence_matrices[k].persistor)

        return HashMatrix(l + L, self.incidence_matrices[k].persistor.get_persistor(0).extend(2))

    def get_next_id(self):
        self.next_id += 1
        return self.next_id

    def __str__(self):
        return f'{self.children}'

def get_cross_persisted_matrices(k):
    persistors = [Persistor() for i in range(k+1)]
    hash_matrix_builders = []

    for i in range(k):
        multi = MultiPersistor()
        multi.add_persistor(persistors[i])
        multi.add_persistor(persistors[i+1])

        hash_matrix_builder = HashMatrixBuilder(multi, header=str(i))
        hash_matrix_builders.append(hash_matrix_builder)

    return hash_matrix_builders

def generate_embedding(s: SimplicalComplex, depth=3, max_dim_per_simplex_dim=10):

    persistor = s.node_persistor
    embedding = None

    _depth = min(s.n, depth)
    if _depth < depth:
        logging.warning('The simplex has less dimesnions than reqested.')
    depth = _depth
    logging.info(f' Generating embedding with \n \
        \t depth: {depth} \n \
        \t max_dimenson_per_simplex: {max_dim_per_simplex_dim}\
        ')

    for i in range(depth):

        laplacian = s.get_k_laplacian(i)
        i_simplices_num = laplacian.size()[0]
        dim = min(max_dim_per_simplex_dim, int(i_simplices_num / 3))
        evals, evecs = torch.lobpcg(laplacian.get_tensor(), k = dim)
        occurences = [0 for _ in range(len(persistor))]

        embedding_slice = torch.zeros(len(persistor), dim)

        for (j, label) in enumerate(laplacian.persistor.get_persistor(0).keys()):
            for letter in label.get_letters():
                index = persistor[Label([letter])]
                embedding_slice[
                    index
                ] += evecs[j]
                occurences[index] += 1


        for (occurence, row) in zip(occurences, embedding_slice):
            if occurence == 0:
                continue
                
            row /= occurence

        if i == 0:
            embedding = embedding_slice
        else:
            embedding = torch.cat((
                embedding,
                embedding_slice),
                1
            )

    return embedding