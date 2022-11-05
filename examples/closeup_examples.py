import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import seaborn as sns
from sklearn.neighbors import NearestNeighbors


cmap = matplotlib.cm.get_cmap('Accent')
colors = [cmap(i) for i in range(10)]
plt.rcParams.update({"font.size": 12})

np.random.seed(1)

n=5
size=128
offsetx = 0.05
offsety = -0.05
r = 1

def plot_words(x, x_query, query, words, name,
                offsetx=0.05,
                offsety=-0.05):
    fig, ax = plt.subplots(figsize=(4, 3))

    ax.scatter(*x_query, color='coral', edgecolors='black', s=size)
    ax.text(x_query[0]+offsetx, x_query[1]-offsety, query, horizontalalignment='center', verticalalignment='center')

    i_color = 0 if name == 'closeup_uk' else 1
    for i, p in enumerate(x):
        ax.scatter(*p, color=colors[i_color], edgecolors='black', s=size)
        con = ConnectionPatch(p, x_query, coordsA='data', coordsB='data', shrinkA=8, shrinkB=8, 
                        edgecolor='black', linewidth=0.5) 
        ax.add_patch(con)
        ax.text(p[0]+offsetx, p[1]+offsety, words[i], horizontalalignment='center', verticalalignment='center')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim((-r*0.9, r*0.9))
    ax.set_ylim((-0.15, r*1.15))
    plt.tight_layout()
    plt.savefig(name+'.pdf')
    plt.close()


neighbors_uk = [
    'bedroom',
    'apartment',
    'suite',
    'storey',
    'fixture'
]

neighbors_us = [
    'truck',
    'suv',
    'flatbed',
    'hummer',
    '4x4'
]

query = 'semi'


# x_uk = np.random.normal((0, 0), (1,1), size=(n, 2))
# x_query = np.mean(x_uk, axis=0)
x_query = np.array([0, 0])

theta_range = np.linspace(-np.pi/4, np.pi/4, 5)
x_uk = np.array([(np.sin(o)*r, np.cos(o)*r) for o in theta_range])

sns.set_style("white")

plot_words(x_uk, x_query, query, neighbors_uk, 'closeup_uk',
            offsetx=0.0, offsety=0.075)

# x_us = np.random.normal((0, 0), (1,1), size=(n, 2))
x_us = np.array([(np.sin(o)*r, np.cos(o)*r) for o in theta_range])

# x_query = np.mean(x_us, axis=0)
plot_words(x_us, x_query, query, neighbors_us, 'closeup_us',
            offsetx=0.0, offsety=0.075)



# fig, ax = plt.subplots()

# ax.scatter(*x_query, color='coral', edgecolors='black', s=size)
# ax.text(x_query[0]+offsetx, x_query[1]+offsety, query)

# for i, p in enumerate(x_uk):
#     ax.scatter(*p, color=colors[0], edgecolors='black', s=size)
#     con = ConnectionPatch(p, x_query, coordsA='data', coordsB='data', shrinkA=8, shrinkB=8, 
#                       edgecolor='black', linewidth=0.5) 
#     ax.add_patch(con)
#     ax.text(p[0]+offsetx, p[1]+offsety, neighbors_uk[i])

# ax.set_xticks([])
# ax.set_yticks([])
# plt.tight_layout()
# plt.savefig('closeup_uk.pdf')

# # emb_a = 'wordvectors/ukus/bnc_pos_lemma.vec'
# # emb_b = 'wordvectors/ukus/coca_pos_lemma.vec'

# # emb_a = 'wordvectors/coca/w_news_2010.vec'
# # emb_b = 'wordvectors/coca/w_mag_2010.vec'

# emb_a = 'wordvectors/ukus/bnc.vec'
# emb_b = 'wordvectors/ukus/coca.vec'

# wv_a = WordVectors(input_file=emb_a, normalized=True)
# wv_b = WordVectors(input_file=emb_b, normalized=True)

# nbrs_a = NearestNeighbors(n_neighbors=20, metric='cosine').fit(wv_a.vectors)
# nbrs_b = NearestNeighbors(n_neighbors=20, metric='cosine').fit(wv_b.vectors)

# # word = 'search'
# word = 'semi'

# x_a = wv_a[word]
# x_b = wv_b[word]
# distances, indices = nbrs_a.kneighbors([x_a])
# print(indices)

# print('----')
# print(' +', word)
# for i in indices[0]:
#     print(wv_a.words[i])

# print("----")
# print(" +", word)
# distances, indices = nbrs_b.kneighbors([x_b])
# for i in indices[0]:
#     print(wv_b.words[i])
