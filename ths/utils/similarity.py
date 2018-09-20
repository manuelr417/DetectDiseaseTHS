import numpy as np

def matrix_cosine_similary(M, N):
    if not (M.shape[1] == N.shape[1]):
        raise ValueError("Arguments cannot be multiplied.")
    # first compute M x N with
    mult_M_N = np.dot(M, N.T)

    #compute norm of M
    sqr_M = M**2
    norm_M = sqr_M.sum(1, keepdims=True) ** 0.5
    #compute norm of N
    sqr_N = N**2
    norm_N = sqr_N.sum(1, keepdims=True) ** 0.5

    # Compute product of norm matrices
    norm_mult = np.dot(norm_M, norm_N.T)

    #now compute reciprocals
    r_norm_mult = norm_mult ** -1

    # finally compute elemetwise product
    # each entry will have the consine similary
    return mult_M_N * r_norm_mult

def distance_similarity_matrix(S):
    I  = np.ones(S.shape)
    return np.average(I - S)