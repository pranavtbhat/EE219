import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
import part1

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

    return df.as_matrix(), R, W

def squared_error(R, W, U, V):
    return np.sum((W * (R - np.dot(U, V))) ** 2)


if __name__ == "__main__":

    data, R_mat, W_mat = load_dataset()
    L = 5
    n_folds = 10

    test_length = len(data) / n_folds
    top_movies_order = []
    kf = KFold(n=len(data), n_folds=10, shuffle=True)

    hcv = []
    mcv = []
    tcv = []
    pcv = []

    for train_index, test_index in kf:
        print "Fold: ", 10 - n_folds + 1
        test_data = data[test_index]

        R_train = W_mat
        W_train = R_mat

        for j in range(test_length):
            W_train[test_data[j][0] - 1, test_data[j][1] - 1] = 0

        U,V = part1.matrix_factorize(R_train, W_train, 100, reg_param=0.01)
        R_predicted = 5 * np.dot(U,V)

        R_predicted[R_train == 0] = -1

        for i in range(max(data[:,0])):
            user_ratings = R_predicted[i]
            top_movies = user_ratings.argsort()[-max(data[:,1]):][::-1]
            top_movies_order.append(top_movies)

        threshold = 3

        hv=[]
        mv=[]
        tv=[]
        pv=[]

        for l in range(1,(L+1)):
            hit = 0
            miss = 0
            total = 0
            precision = 0
            for i in range(max(data[:,0])):
                rec_indices = R_predicted[i,0:l]
                for j in range(len(rec_indices)):
                    rating = R_predicted[i][rec_indices[j]]
                    if (rating < 0):
                        continue
                    if (rating > threshold):
                        hit = hit + 1
                        total = total + 1
                        precision += 1
                    else:
                        miss = miss + 1
                        total = total + 1

            pv.append(precision/float(total))
            hv.append(hit)
            tv.append(total)
            mv.append(miss)

        hcv.append(hv)
        mcv.append(mv)
        tcv.append(tv)
        pcv.append(pv)
        n_folds -= 1

    precision = np.sum(pcv,axis=0)
    hits = np.sum(hcv,axis=0)
    miss = np.sum(mcv,axis=0)
    total = np.sum(tcv,axis=0)

    hits = hits / (total.astype(float))
    miss = miss / (total.astype(float))
    precision = precision / 10.0

    print "Hits ", hits
    print "Misses", miss
    print "Precision : ", precision

    plt.figure(1)
    plt.ylabel('Hit rate')
    plt.xlabel('L')
    plt.title('Hit rate vs L')
    plt.scatter(range(1,(L+1)), hits, s=60, marker='o')
    plt.plot(range(1,(L+1)),hits)
    plt.savefig("plots/Hit vs L.png",format='png')
    plt.clf()

    plt.figure(1)
    plt.ylabel('False alarm')
    plt.xlabel('L')
    plt.title('False Alarm vs L')
    plt.scatter(range(1,(L+1)), miss, s=60, marker='o')
    plt.plot(range(1,(L+1)),miss)
    plt.savefig("plots/False Alarm vs L.png",format='png')
    plt.clf()

    plt.figure(1)
    plt.ylabel('Hit rate')
    plt.xlabel('False Alarm')
    plt.title('Hit rate vs False Alarm')
    plt.scatter(miss, hits, s=60, marker='o')
    plt.plot(miss,hits)
    plt.savefig("plots/Hit rate vs False Alarm.png",format='png')
    plt.clf()
