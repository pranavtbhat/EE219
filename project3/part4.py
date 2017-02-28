import numpy as np
import matplotlib.pyplot as plt
import part1


def plot_ROC(predicted, actual, k, alpha):

    tp = 0
    fp = 0
    fn = 0

    threshold_value = np.arange(1,6,1)
    precision = np.zeros(len(threshold_value))
    recall = np.zeros(len(threshold_value))

    for x, t in enumerate(threshold_value):
        tp = np.sum(actual[predicted >= t] >= t)
        fp = np.sum(actual[predicted >= t] < t)
        fn = np.sum(actual[predicted < t] >= t)

        precision[x] = tp / float(tp+fp)
        recall[x] = tp / float(tp+fn)

    plt.figure(1)
    plt.ylabel('Recall')
    plt.xlabel('Precision')
    plt.title('Plot for k={0} and lambda={1}'.format(k ,alpha))
    plt.scatter(precision, recall, s=60, marker='o')
    plt.plot(precision,recall)
    plt.savefig("plots/ROC"+str(k)+ "_" +str(alpha)+".png",format='png')
    plt.clf()

if __name__ == "__main__":
    R, W = part1.load_dataset()

    for k in [10, 50, 100]:
        print "Setting k = ", k
        U, V = part1.matrix_factorize(W, R, k, reg_param=0, num_iterations=200)

    for k in [10, 50, 100]:
        print "Setting k = ", k
        for alpha in [0.01, 0.1, 1]:
            print "Setting alpha = ", alpha
            U, V = part1.matrix_factorize(R, W, k, reg_param=alpha)
            plot_ROC(np.dot(U, V), R, k, alpha)
