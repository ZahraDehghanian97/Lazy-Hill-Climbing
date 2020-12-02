import matplotlib.pyplot as plt
import numpy as np
import time
import scipy.io


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
        result[int(text[0]), int(text[1])] = abs(float(text[2]))
    return result


# this function build n realization of probabilistic graph
def build_probable_matrices(adjacency_matrix, num_realization, p):
    list_m = []
    for x in range(num_realization):
        temp = np.array(adjacency_matrix)
        for i in range(num_node):
            indexes = np.nonzero(temp[i])
            for j in indexes[0]:
                temp[i, j] = np.random.uniform(0, 1, 1)[0] < p
                temp[j,i] = temp[i,j]
        list_m.append(temp)
    return list_m


def get_neighbor(g, node):
    return np.nonzero(g[node])[0]


def IC2(list_g, S):
    score = 0
    for g in list_g:
        for s in S:
            score += np.count_nonzero(g[s])
    score /= len(list_g)
    return score


def IC(list_g, S):
    spread = []
    for i in range(len(list_g)):
        g = list_g[i]
        new_active, A = S[:], S[:]
        while new_active:
            new_ones = []
            for node in new_active:
                new_ones += list(set(get_neighbor(g, node)))
            new_active = list(set(new_ones) - set(A))
            A += new_active
        spread.append(len(A))
    return np.mean(spread)


def greedy_hill_climbing(g, k):
    S, spread, timelapse, start_time = [], [], [], time.time()
    # Find k nodes with largest marginal gain
    kprime = min(k, num_node)
    for o in range(kprime):
        best_spread = 0
        for j in set(range(num_node)) - set(S):
            s = IC(g, S + [j])
            if s > best_spread:
                best_spread, node = s, j
        S.append(node)
        print("find " + str(o + 1) + "nd member of S = "+str(node))
        spread.append(best_spread)
        timelapse.append(time.time() - start_time)
    return S, spread, timelapse


def lazy_hill_climbing(g, k):
    start_time = time.time()
    marg_gain = [IC(g, [node]) for node in range(num_node)]
    Q = sorted(zip(range(num_node), marg_gain), key=lambda x: x[1], reverse=True)
    S, SPREAD = [Q[0][0]], [Q[0][1]]
    Q, timelapse = Q[1:], [time.time() - start_time]
    print("find 1st member of S")
    kprime = min(k, num_node)
    for o in range(1, kprime):
        check = False
        while not check:
            current = Q[0][0]
            Q[0] = (current, IC(g, S + [current]))
            Q = sorted(Q, key=lambda x: x[1], reverse=True)
            check = (Q[0][0] == current)
        S.append(Q[0][0])
        SPREAD.append(Q[0][1])
        timelapse.append(time.time() - start_time)
        print("find " + str(o + 1) + "nd member of S = "+str(Q[0][0]))
        Q = Q[1:]
    return S, SPREAD, timelapse


# load file
input_file = 'facebook101_princton_weighted.mat'
txt_input_file = 'dataset.txt'
# change_MAT_to_TXT(input_file, txt_input_file)
num_node = 6596
adjacency_matrix = build_matrix(txt_input_file, num_node)
print("read input file and convert to matrix")

# genetate realization
num_realization = 2
list_realization = build_probable_matrices(adjacency_matrix, num_realization, p=0.1)
print("generate " + str(num_realization) + " realization successfully")

# # fast test
# list_realization = [[[0, 1, 1], [1, 0, 1], [1, 1, 0]], [[0, 1, 0], [1, 0, 1], [0, 1, 0]]]
# adjacency_matrix = [[0,0.5,1],[0.5,0,0.25],[1,0.25,0]]
# num_node = 3
# num_realization = 2

# Run algorithms
size_S = 3
print("start running lazy_hill_climbing")
lazy_hill_climbing_output = lazy_hill_climbing(list_realization, size_S)
print("lazy_hill_climbing output:   " + str(lazy_hill_climbing_output[0]))

print("start running greedy_hill_climbing")
greedy_hill_climbing_output = greedy_hill_climbing(list_realization, size_S)
print("greedy_hill_climbing output: " + str(greedy_hill_climbing_output[0]))
#
# Plot settings
plt.rcParams['figure.figsize'] = (9, 6)
plt.rcParams['lines.linewidth'] = 4
plt.rcParams['xtick.bottom'] = False
plt.rcParams['ytick.left'] = False

# Plot Computation Time
plt.subplots()
plt.plot(range(1, len(greedy_hill_climbing_output[2]) + 1), greedy_hill_climbing_output[2], label="greedy_hill_climbing", color="#FBB4AE")
plt.plot(range(1, len(lazy_hill_climbing_output[2]) + 1), lazy_hill_climbing_output[2], label="lazy_hill_climbing", color="#B3CDE3")
plt.ylabel('Computation Time (Seconds)')
plt.xlabel('Size of Seed Set')
plt.title('Computation Time')
plt.legend(loc=2)

# Plot Expected Spread by Seed Set Size
plt.subplots()
plt.plot(range(1, len(greedy_hill_climbing_output[1]) + 1), greedy_hill_climbing_output[1], label="greedy hill climbing", color="#FBB4AE")
plt.plot(range(1, len(lazy_hill_climbing_output[1]) + 1), lazy_hill_climbing_output[1], label="lazy hill climbing", color="#B3CDE3")
plt.xlabel('Size of Seed Set')
plt.ylabel('Expected Spread')
plt.title('Expected Spread')
plt.legend(loc=2)
plt.show()
