import random

import scipy.io
import numpy as np


# this function convert MAT file to txt
def change_MAT_to_TXT(input_file, output_file):
    mat = scipy.io.loadmat(input_file)
    A = mat.__getitem__('A')
    file = open(output_file, 'w')
    for i in range(A.shape[0]):
        for j in A[i].nonzero()[1]:
            file.write(str(i) + ' ' + str(j) + ' ' + str(A[i, j]) + '\n')
    file.close()


# this function build adjeceny matrix from txt file
def build_matrix(dataset, size_matrix):
    result = np.zeros([size_matrix, size_matrix])
    A = open(dataset, encoding='utf-8')
    for line in A:
        text = line.split()
        result[int(text[0]), int(text[1])] = 1  # float(text[2])
    return result


# this function build n realization of probabilistic graph
def build_probable_matrixs(n, p):
    list_m = []
    for i in range(n):
        temp = np.array(adjacency_matrix)
        for i in range(len_matrix):
            indexes = np.nonzero(temp[i])
            for j in indexes[0]:
                temp[i, j] = random.choices([0, 1], [1 - p[i, j], p[i, j]])[0]
        list_m.append(temp)
    return list_m


def score_add(node, S, A):
    score = 0
    for j in range(len_matrix):
        if  A[node, j] > 0 and (j not in S) :
            score += 1
    return score


input_file = 'facebook101_princton_weighted.mat'
txt_input_file = 'dataset.txt'
# change_MAT_to_TXT(input_file, txt_input_file)
adjacency_matrix = build_matrix(txt_input_file, 6596)
len_matrix = len(adjacency_matrix)

# Generate n realization of probabilistic Graph
print("Generate realization ")
n = 10  # number of realization
p = np.full((len_matrix, len_matrix), 0.5)  # probability matrix of activation
list_matrices = build_probable_matrixs(n, p)


# Compute S set with Greedy hill climbing
print("run Greedy hill climbing Algorithm on realizations")
size_s = 10
S = []
f_s = []
for i in range(size_s):
    print("start finding "+str(len(S)+1)+"th member of S")
    f_si = np.zeros(len_matrix)
    c = 0
    for A in list_matrices:
        c+=1
        for node in range(len_matrix):
            if node not in S :
                f_si[node] += score_add(node, S, A)
        print("compute f_si for realization number "+str(c))
    score_max = max(f_si)
    node = np.where(f_si == score_max)[0][0]
    S.append(node)
    score_max /= n
    f_s.append(score_max/n)
    print("add node number "+str(node)+ " with mean score = "+str(round(score_max,2)))
print("S set = "+str(S))
print("mean score of each node = "+str(f_s))

