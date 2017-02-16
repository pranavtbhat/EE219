from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
from collections import Counter


def fetch_all_categories():
    return [
       "comp.graphics",
       "comp.os.ms-windows.misc",
       "comp.sys.ibm.pc.hardware",
       "comp.sys.mac.hardware"
    ] + [
       "rec.autos",
       "rec.motorcycles",
       "rec.sport.baseball",
       "rec.sport.hockey"
    ]

def fetch_train(categories):
    return fetch_20newsgroups(
        subset = 'train',
        categories = categories,
        shuffle=True,
        random_state = 42
    )

def fetch_test(categories):
    return fetch_20newsgroups(
        subset = 'test',
        categories = categories,
        shuffle=True,
        random_state = 42
    )

if __name__ == "__main__":
    train = fetch_train(fetch_all_categories())
    topic_names = train.target_names

    cc = Counter(train.target)
    freqs = [cc[i] for i in cc]

    plt.bar(range(0, 8), freqs, 1/1.5, color='blue')
    plt.xticks(range(0, 8), train.target_names, rotation=90)

    plt.subplots_adjust(bottom=0.45)
    plt.xlabel("Categories")
    plt.ylabel("Frequencies")

    plt.savefig('plots/histogram.png', format='png')
    plt.show()
    num_ct = sum(freqs[0:4])
    num_rec = sum(freqs[4:8])


    print "Number of documents in the Computer Technology class is ", num_ct
    print "Number of documents in the Recreational activity class is", num_rec
