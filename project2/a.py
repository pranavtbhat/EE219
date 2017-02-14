from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
from collections import Counter

comp_tech = [
    "comp.graphics",
    "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware"
]

rec_act = [
    "rec.autos",
    "rec.motorcycles",
    "rec.sport.baseball",
    "rec.sport.hockey"
]

train = fetch_20newsgroups(
    subset = 'train',
    categories = comp_tech + rec_act,
    shuffle=True,
    random_state=23
)
topic_names = train.target_names

cc = Counter(train.target)
freqs = [cc[i] for i in cc]

plt.bar(range(1, 9), freqs, 1/1.5, color='blue')
plt.xticks(range(1, 9), train.target_names, rotation=90)

plt.subplots_adjust(bottom=0.45)
plt.xlabel("Categories")
plt.ylabel("Frequencies")

plt.savefig('plots/histogram.png', format='png')
plt.show()
num_ct = sum(freqs[0:3])
num_rec = sum(freqs[4:7])


print "Number of documents in the Computer Technology class is ", num_ct
print "Number of documents in the Recreational activity class is", num_rec
