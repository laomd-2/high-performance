import scipy.sparse
import time
from collections import Counter


def output(matrix):
    for row in matrix.toarray():
        for x in row:
            print("%.6f" % x, end=' ')
        print()


if __name__ == '__main__':
    m2 = n2 = 64
    matrix: scipy.sparse.csr_matrix = scipy.sparse.rand(m2, n2, 0.1, 'csr', float, random_state=0)
    matrixb: scipy.sparse.csr_matrix = scipy.sparse.rand(m2, n2, 0.1, 'csr', float, random_state=1)
    # print(matrix.toarray()[0:4, 0: 4] * matrixb.toarray()[4:8, 4: 8])
    print(matrix * matrixb)
    # row, col = 0, 32746
    # print((matrix * matrixb).getrow(0))

    # i = 0
    # for x, y in zip(matrix.getrow(row).toarray()[0], matrixb.getcol(col).toarray()[:, 0]):
    #     if x != 0 and y != 0:
    #         print(i, x, y)
    #     i += 1
    # res = ((matrix * matrixb).toarray()[0])
    # for i, x in enumerate(res):
    #     if x != 0:
    #         print(0, i, x)
    which = matrix
    nonzero = which.nonzero()
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
        for x in which.data:
            print('%.6f' % x, end=' ', file=out)
        print(file=out)
        for x in nonzero[1]:
            print(x, end=' ', file=out)
        print(file=out)
        for x in row_indices:
            print(x, end=' ', file=out)
    # print(matrix.toarray())
    # start = time.time()
    # c = matrix * matrix
    # print(time.time() - start)
