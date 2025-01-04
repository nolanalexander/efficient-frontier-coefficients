import pandas as pd
import numpy as np

       
def calc_transition_matrix(transitions):
    num_states = max(transitions)
    matrix = pd.DataFrame(0, columns=range(1, num_states+1), index=range(1, num_states+1))
    for (i,j) in zip(np.array(transitions), np.array(transitions[1:])):
        matrix[i][j] += 1
    matrix = matrix.divide(matrix.sum(axis=1), axis=0).replace(np.nan, 0)
    return matrix

def find_transition_state(transition_matrix, cur_state, num_samps=1):
    pdf = transition_matrix.loc[cur_state]
    cdf = pdf.cumsum()
    rand_nums = np.random.uniform(0, 1, num_samps)
    return np.searchsorted(cdf, rand_nums) + 1

def calc_steady_state_prob(transition_matrix):
    dim = transition_matrix.shape[0]
    q = (transition_matrix - np.eye(dim))
    ones = np.ones(dim)
    q = np.c_[q,ones]
    QTQ = np.dot(q, q.T)
    bQT = np.ones(dim)
    return np.linalg.solve(QTQ,bQT)


