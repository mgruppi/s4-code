"""
Runs semantic change experiment on arxiv data.
Saves a file un results/arxiv/ containing the semantic shift scores for
each alignment method (Global, Noise-Aware, S4) in order to compare them.
"""

from WordVectors import WordVectors, intersection
from alignment import align
from noise_aware import noise_aware
from s4 import s4

from scipy.spatial.distance import cosine, euclidean
import numpy as np
import os
import argparse

from collections import defaultdict


def align_wordvectors(*wvs, method="global"):
    target = wvs[0]
    aligned = [target]
    for wv in wvs[1:]:
        if method == "global":
            wv, tg, Q = align(wv, target)
        elif method == "noise_aware":
            Q, alpha, l, k = noise_aware(wv.vectors, target.vectors)
            wv.vectors = np.dot(wv.vectors,Q)
        aligned.append(wv)
    return aligned


def distribution_of_change(*wvs, metric="euclidean"):
    """
    Gets distribution of change per word across input WordVectors list wvs.
    Assumes the WordVectors in wvs have been previously aligned to the same reference point
    (E.g.: align all to wvs[0]).
    Arguments:
            wvs - list of WordVectors objects
    Returns:
            d   - array of N elements with the mean cosine distance across the aligned WordVectors
                    (N is the size of the common vocabulary)
    """

    d = np.zeros((len(wvs[0])))
    for i, w in enumerate(wvs[0].words):
        # Compute mean vector
        v_mean = np.mean([wv[w] for wv in wvs], axis=0)
        # Compute distances to the mean
        if metric == "euclidean":
            distances = [np.linalg.norm(v_mean-wv[w])**2 for wv in wvs]
        elif metric == "cosine":
            distances = [cosine(v_mean, wv[w]) for wv in wvs]
        # distances = [cosine(v_mean, wv[w]) for wv in wvs]
        mean_d = np.mean(distances)
        d[i] = mean_d
    return d


def print_table(d, words, n=20):
    """
    Prints table of stable and unstable words in the following format:
    <stable words> | <unstable words>
    Arguments:
                d       - distance distribution
                words   - list of words - indices of d and words must match
                n       - number of rows in the table
    """
    print("-"*20)
    print("%15s\t%15s" % ("stable", "unstable"))
    indices = np.argsort(d)
    for i in range(n):
        print("%15s\t%15s"
              % (words[indices[i]], words[indices[-i-1]]))
    print("-"*20)

def main():
    """
    The following experiments are available:
        - Find most stable words in each ArXiv category (cs, math, cond-mat, physics)
        - Find most unstable (changed) words in earch category
        - Finds stable/unstable words across categories
        - Using different alignment strategies
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("cat1", type=str, help="Name of first arXiv category")
    parser.add_argument("cat2", type=str, help="Name of second arXiv category")

    args = parser.parse_args()

    cat1 = args.cat1
    cat2 = args.cat2

    cat1_name = cat1.split("_")[2].rstrip(".vec")
    cat2_name = cat2.split("_")[2].rstrip(".vec")

    path_out = "results/arxiv/"

    wva = WordVectors(input_file=cat1)
    wvb = WordVectors(input_file=cat2)
    wva, wvb = intersection(wva, wvb)
    wva, wvb, Q = align(wva, wvb)
    words = wva.words

    print("-- Common vocab", len(words))
    # each column of this matrix will store a set of results for a method
    out_grid = np.zeros((len(words), 5))

    d = distribution_of_change(wva, wvb)
    print("====== GLOBAL")
    print("=> landmarks", len(wva.words))
    print_table(d, wva.words)
    out_grid[:, 0] = d  # add first column

    print("====== Noise Aware")

    Q, alpha, landmarks, noisy = noise_aware(wva.vectors, wvb.vectors)
    wva, wvb, Q = align(wva, wvb, anchor_words=landmarks)
    print("=> landmarks", len(landmarks))
    d = distribution_of_change(wva, wvb)
    print_table(d, wva.words)
    out_grid[:, 1] = d  # add new column

    print("===== SELF")
    landmarks, nonl, Q = s4(wva, wvb, iters=100, verbose=1)
    wva, wvb, Q = align(wva, wvb, anchor_words=landmarks)
    d = distribution_of_change(wva, wvb)
    print_table(d, wva.words)
    out_grid[:, 2] = d  # last column

    # WRITE-OUT
    with open(os.path.join(path_out, "%s-%s.csv" % (cat1_name, cat2_name)), "w") as fout:
        fout.write("word,global,noise-aware,self,top,bot\n")
        for i, w in enumerate(words):
            fout.write("%s,%.3f,%.3f,%.3f,%.3f,%.3f\n" % (w, out_grid[i][0], out_grid[i][1],
                        out_grid[i][2], out_grid[i][3], out_grid[i][4]))


if __name__ == "__main__":
    main()
