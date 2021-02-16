"""
Produce ranking correlation experiment between arXiv alignments accoring
to ranked files in results/arxiv/
Performs spearman rho's correlation at varying top-K most changed words.
"""
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import argparse
import numpy as np
from itertools import zip_longest


def compare_rankings(a, b, d_a, d_b, k):
    """
    Compare the rankings of two arg sorted lists a and b using
    scores d_a, d_b.
    The union of a and b is used, disjoint elements will be treated as if they
    were at the bottom of the other list.
    The top-k indices are chosen from a and b, and then combined.
    Arguments:
        a   - argsorted list of words (indices)
        b   - argsorted list of words (indices)
        d_a - scores of rank a
        d_b - scores of rank b
        k   - compare words at top-K
    Returns:
        Spearman rho for the disjoint rankings
    """
    indices = set.union(set(a[:k]), set(b[:k]))

    r_a = [d_a[i] for i in indices]
    r_b = [d_b[i] for i in indices]

    rho, pvalue = spearmanr(r_a, r_b)
    return rho


def main():
    """
    Do ranking of disjoint lists
    We'll treat disjoint elements as if they were at the bottom of
    the list (which they actually are if K is increased)
    To that end, we take the union of the words in each top-k
    and apply spearman rho to the resulting list
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("input")

    args = parser.parse_args()

    plt.style.use("seaborn")



    with open(args.input) as fin:
        fin.readline()
        data = map(lambda s: s.strip().split(",", 1), fin.readlines())
    words, d = zip(*data)
    d = np.array([v.split(",") for v in d], dtype=float)

    descending = True

    # Sorted indices of words for each method, large first
    sort_global = np.argsort(d[:, 0])
    sort_noise = np.argsort(d[:, 1])
    sort_self = np.argsort(d[:, 2])

    if descending:
        sort_global = sort_global[::-1]
        sort_noise = sort_noise[::-1]
        sort_self = sort_self[::-1]

    k_values = np.arange(10, 510, 10)
    marker_gap = 10  # every N points, plot a marker on the lines

    # These lists hold the ranking correl. values for each top-k iteration
    gn_list = list()
    gs_list = list()
    ns_list = list()
    for k in k_values:
        # print("- k", k)

        # Global and Noise-Aware
        r_gn = compare_rankings(sort_global, sort_noise, d[:, 0], d[:, 1], k)
        r_gs = compare_rankings(sort_global, sort_self, d[:, 0], d[:, 2], k)
        r_ns = compare_rankings(sort_noise, sort_self, d[:, 1], d[:, 2], k)

        # print(round(r_gn, 2), round(r_gs, 2))

        gn_list.append(r_gn)
        gs_list.append(r_gs)
        ns_list.append(r_ns)

    plt.plot(k_values, gn_list, label="Global x Noise", c="#0073b7")
    plt.plot(k_values, gs_list, label="Global x S4-A", c="#e74c3c")
    plt.plot(k_values, ns_list, label="Noise x S4-A", c="#7474c1")

    # Plot markers

    plt.tick_params(axis='both', which='major', labelsize=18)


    for i in range(0, len(k_values), marker_gap):
        plt.scatter(k_values[i], gn_list[i], marker="o", c="#0073b7")
        plt.scatter(k_values[i], gs_list[i], marker="s", c="#e74c3c")
        plt.scatter(k_values[i], ns_list[i], marker="d", c="#7474c1")

    plt.legend(fontsize=18)
    plt.xlabel("Top-K", fontsize=18)
    plt.ylabel("Spearman rho", fontsize=18)
    plt.tight_layout()
    # plt.show()
    plt.savefig("results/arxiv/arxiv_ranking.pdf", format="pdf")

    N = 50  # top shifted
    K = 50  # top common
    set_global = set(sort_global[:N])
    set_noise = set(sort_noise[:N])
    set_self = set(sort_self[:N])

    inters = set.intersection(set_global, set_noise, set_self)

    isect = [words[i] for i in inters]
    isect = sorted(isect[:K])
    isect_left = isect[:K//2]
    isect_right = isect[K//2:K]
    words_g = sorted([words[i] for i in (set_global-set_self)])
    words_n = sorted([words[i] for i in set_noise-set.union(set_global, set_self)])
    words_s = sorted([words[i] for i in set_self - set.union(set_global,set_noise)])

    # Table in LaTeX format
    # for a,b,c,d,e in zip_longest(words_g, words_n, words_s, isect_left, isect_right, fillvalue=""):
    #     print("%s & %s & %s & %s & %s \\\\" % (a,b,c,d,e))
    # # Table in readable format
    # print("%15s %15s %15s | %15s" % ("Global", "Noise-Aware", "S4-A", "Common"))
    # print("-"*80)
    # for a,b,c,d,e in zip_longest(words_g, words_n, words_s, isect_left, isect_right, fillvalue=""):
    #     print("%15s %15s %15s %15s %15s " % (a,b,c,d,e))
    # Table in Markdown format
    print("|Global|Noise-Aware|S4-A|Common| |")
    print("|------|-----------|----|------|-|")
    for a,b,c,d,e in zip_longest(words_g, words_n, words_s, isect_left, isect_right, fillvalue=""):
        print("|%s|%s|%s|%s|%s|" % (a,b,c,d,e))



if __name__ == "__main__":
    main()
