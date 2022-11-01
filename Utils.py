import sys


def print_matrix(Title, M):
    print(Title)
    for row in M:
        print([round(x, 3) + 0 for x in row])


def print_matrices(Action, M1, M2, markov_terms, to_remove_idx=None):
    print(Action)
    # print(Title1, '\t' * int(len(M1) / 2) + "\t" * len(M1), Title2)
    for i in range(len(M1)):
        if i == to_remove_idx:
            continue
        row1 = " + ".join(['{0:-.3f}'.format(x[0]) + " * " + str(x[1]) for x in zip(M1[i], markov_terms)])
        row2 = "".join(['{0:.3f}'.format(x) for x in M2[i]])
        print(row1, '=', row2)


def zeros_matrix(rows, cols):
    A = []
    for i in range(rows):
        A.append([])
        for j in range(cols):
            A[-1].append(0.0)

    return A


def copy_matrix(M):
    rows = len(M)
    cols = len(M[0])

    MC = zeros_matrix(rows, cols)

    for i in range(rows):
        for j in range(cols):
            MC[i][j] = M[i][j]

    return MC


def matrix_multiply(A, B):
    rowsA = len(A)
    colsA = len(A[0])

    rowsB = len(B)
    colsB = len(B[0])

    if colsA != rowsB:
        print('Number of A columns must equal number of B rows.')
        sys.exit()

    C = zeros_matrix(rowsA, colsB)

    for i in range(rowsA):
        for j in range(colsB):
            total = 0
            for ii in range(colsA):
                total += A[i][ii] * B[ii][j]
            C[i][j] = total

    return C