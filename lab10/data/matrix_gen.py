import scipy.sparse
import time
from collections import Counter


def output(matrix):
    for row in matrix.toarray():
        for x in row:
            print("%.6f" % x, end=' ')
        print()


if __name__ == '__main__':
    m1 = n1 = 1024*8
    m2 = n2 = 4096
    matrix: scipy.sparse.csr_matrix = scipy.sparse.rand(m2, n2, 0.0015, 'csr', float, random_state=0)
    nonzero = matrix.nonzero()
    cnt = Counter()
    for row in nonzero[0]:
        cnt[row] += 1
    index = 0
    row_indices = []
    for i in range(m2):
        row_indices.append(index + cnt[i])
        index += cnt[i]
    row_indices.append(len(nonzero[1]) + 1)
    with open('csr_sparse' + str(m2) + '.mtx', 'w') as out:
        print(len(row_indices) - 1, len(nonzero[0]), file=out)
        for x in matrix.data:
            print('%.6f' % x, end=' ', file=out)
        print(file=out)
        for x in nonzero[1]:
            print(x, end=' ', file=out)
        print(file=out)
        for x in row_indices:
            print(x, end=' ', file=out)
    print(matrix.toarray())
    # start = time.time()
    # matrix * matrix
    # print(time.time() - start)
