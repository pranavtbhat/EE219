import pandas as pd
import numpy as np
from scipy import linalg

def load_dataset():
    df = pd.read_csv(
        'ml-100k/u.data',
        delimiter='\t',
        names = ['user_id', 'item_id', 'rating', 'timestamp'],
        header=0
    )

    R = df.pivot_table(
        index=['user_id'],
        columns=['item_id'],
        values='rating',
        fill_value=0
    ).values

    W = R.copy()
    W[W > 0] = 1

    return R, W

def squared_error(R, W, U, V):
    return np.sum((W * (R - np.dot(U, V))) ** 2)

def matrix_factorize(R, W, k, reg_param=0, num_iterations=100):
    eps = 1e-5

    m, n = R.shape
    U = np.maximum(eps, 5 * np.random.rand(m, k))
    V = np.maximum(eps, linalg.lstsq(U, R)[0])

    WR = W * R

    for i in range(num_iterations):
        # Solve for U using V as a constraint
        top = np.dot(WR, V.T)
        bottom = np.add(
            np.dot(W * np.dot(U, V), V.T),
            reg_param * U
        ) + eps
        U = np.maximum(eps, U * top / bottom)

        # Solve for V using U as a constraint
        top = np.dot(U.T, WR)
        bottom = np.add(
            np.dot(U.T, W * np.dot(U, V)),
            reg_param * V
        ) + eps
        V = np.maximum(eps, V * top / bottom)

    print "Matrix was factorized with MSQE ", squared_error(R, W, U, V)

    return U, V


if __name__ == "__main__":
    R, W = load_dataset()

    for k in [10, 50, 100]:
        print "Setting k = ", k
        U, V = matrix_factorize(R, W, k)
