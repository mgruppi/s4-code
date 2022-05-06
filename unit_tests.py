from s4 import s4, inject_change_single
from s4_torch import S4Network
from alignment import align
from WordVectors import WordVectors, intersection
from scipy.spatial.distance import cosine
from sklearn.metrics import pairwise_distances
import numpy as np
import time


def inject_change_test(wv1, wv2):
    wva, wvb = intersection(wv1, wv2)
    wva, wvb, q = align(wva, wvb)

    n_runs = 10
    n_targets = 100
    r = 1
    t_init = time.time()
    targets = np.random.choice(wva.words, n_targets)

    choice_methods = ['random', 'close', 'far']
    d_matrix = pairwise_distances(wvb.vectors, metric='cosine', n_jobs=-1)

    d_cos = dict()
    times = dict()
    for c in choice_methods:
        d_cos[c] = list()

    for c in choice_methods:
        t0 = time.time()
        print(c)
        for t in targets:
            v = inject_change_single(wvb, t, wva.words, wva[t], r, choice_method=c, distances=d_matrix[wvb.word_id[t]])
            d_cos[c].append(cosine(v, wvb[t]))
        times[c] = time.time() - t0
    
    print("-- Distribution of cosine distances after perturbation")
    for c in choice_methods:
        print("  +", c, np.mean(d_cos[c]), "+-", np.std(d_cos[c]), "(%.2f s)" % (times[c]))
    
    print("Total time: %.2f s" % (time.time() - t_init))
    print("---"*10)


if __name__ == "__main__":
    np.random.seed(0)
    normalized = True
    wv1 = WordVectors(input_file='wordvectors/semeval/english-corpus1.vec', normalized=normalized)
    wv2 = WordVectors(input_file='wordvectors/semeval/english-corpus2.vec', normalized=normalized)

    inject_change_test(wv1, wv2)


    