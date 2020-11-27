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


input_file = 'facebook101_princton_weighted.mat'
txt_input_file = 'dataset.txt'
# change_MAT_to_TXT(input_file, txt_input_file)
adjacency_matrix = build_matrix(txt_input_file, 6596)
len_matrix = len(adjacency_matrix)

# Make n realization of probabilistic Graph
n = 2  # number of realization
p = np.full((len_matrix,len_matrix),0.5)  # probability matrix of activation
list_matrices = build_probable_matrixs(n, p)
print(np.count_nonzero(adjacency_matrix))
print(np.count_nonzero(list_matrices[0]))
print(np.count_nonzero(list_matrices[1]))
