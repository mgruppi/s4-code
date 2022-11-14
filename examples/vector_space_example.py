import seaborn as sns
import matplotlib.pyplot as plt
from WordVectors import WordVectors
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import numpy as np


input = "wordvectors/ukus/coca.vec"

plt.rcParams.update({"font.size": 12})
sns.set_style("whitegrid")

wv = WordVectors(input_file='wordvectors/ukus/coca.vec')

query = 'computer'
n = 50

nbrs = NearestNeighbors(metric='cosine', n_neighbors=n).fit(wv.vectors)

v = wv[query]
distances, indices = nbrs.kneighbors([v])

query_idx = wv.word_id[query]
words = [wv.words[i] for i in indices[0]]

print(words)

vectors = np.array([wv[w] for w in words])

x = TSNE(n_components=2, metric='cosine').fit_transform(vectors)
# x = PCA(n_components=2).fit_transform(vectors)

fig, ax = plt.subplots(figsize=(10, 10))
for i, w in enumerate(words):
    if w != query:
        ax.text(x[i][0], x[i][1], w)
        ax.scatter(x[i][0], x[i][1], alpha=0)
    else:
        ax.text(x[i][0], x[i][1], w, fontweight='bold', fontsize='large')
        ax.scatter(x[i][0], x[i][1], alpha=0)

plt.savefig('examples/vector_space.pdf')
