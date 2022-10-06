from WordVectors import WordVectors
import numpy as np
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import pearsonr, spearmanr
import os


def load_word_counts(path):
    """
    Loads a word counts file into a dictionary.

    Args:
        path(str) : Input file.
    
    Returns:
        counts(dict[str->int]) : Word counts.
    """
    counts = dict()
    with open(path) as fin:
        for line in fin:
            t = line.strip().split(" ", 1)
            if len(t) < 2:
                continue
            w, c = t
            counts[w] = int(c)
    return counts


def get_vector_norms(wv):
    """
    Returns the vector norms of each word in `wv`.

    Args:
        wv(WordVectors) : Input word vectors.
    
    Returns:
        norms(dict[str->float]) : Dictionary of vector norms.
    """

    norms = dict()

    for word in wv.words:
        norms[word] = np.linalg.norm(wv[word])

    return norms


def get_distributions(counts, norms, filter=None, samples=1000):
    """
    Returns the distributions of counts and norms.
    Selects from `counts` only those words that appear in `norms`.
    If `filter` is passed, filter only those pos tags. Otherwise, select every word in common.

    Args:
        counts(dict[str->int]) : Word count dicionary.
        norms(dict[str->float]) : Vector norm dictionary.
        filter(list[str]) : List of pos tags to include. If `None`, do not apply filtering.
        samples(int) : The number of samples to keep.

    Returns:
        wds(list[str]) : List of words.
        cts(list[int]) : List of counts.
        nrms(list[float]) : List of norms.
    """

    wds, cts, nrms = list(), list(), list()

    for w in norms:
        pos_tag = nltk.pos_tag([w])[0][1]
        if w in counts and (filter is None or pos_tag in filter):
            if counts[w] < 100:
                continue
            wds.append(w)
            cts.append(counts[w])
            nrms.append(norms[w])
    
    sample_indices = np.random.choice(range(len(wds)), size=samples)
    wds = [wds[i] for i in sample_indices]
    cts = [cts[i] for i in sample_indices]
    nrms = [nrms[i] for i in sample_indices]

    return wds, cts, nrms


if __name__ == "__main__":

    matplotlib.rcParams.update({"font.size": 12})
    sns.set_style("whitegrid")
    output_dir = "results/frequency-norm-dist"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cmap = matplotlib.cm.get_cmap('Accent')
    # colors = ["#2E53A3", "#F06C51", "#CCF089", "#789EF0", "#6EA30C", "#F0AD4A"]
    # colors = ["#112663", "#431163", "#63113D", "#632B11", "#546311", "#116314", "#11635A"]
    colors = ["#000080", "#6E0080", "#800025", "#804900", "#498000", "#008025", "#006E80"]
    colors = [cmap(i) for i in range(10)]

    datasets = {
        "English (old)" : {
            "wordvectors": "wordvectors/semeval/english-corpus1.vec",
            "wordcounts": "results/word_counts/english-corpus1.txt",
            "color": colors[0]
        },
        "English (modern)": {
            "wordvectors": "wordvectors/semeval/english-corpus2.vec",
            "wordcounts": "results/word_counts/english-corpus2.txt",
            "color": colors[0]
        },
        "German (old)": {
            "wordvectors": "wordvectors/semeval/german-corpus1.vec",
            "wordcounts": "results/word_counts/german-old-dta.txt",
            "color": colors[1]
        },
        "German (modern)": {
            "wordvectors": "wordvectors/semeval/german-corpus2.vec",
            "wordcounts": "results/word_counts/german-modern-bznd.txt",
            "color": colors[1]
        },
        "Latin (old)": {
            "wordvectors": "wordvectors/semeval/latin-corpus1.vec",
            "wordcounts": "results/word_counts/latin-LatinISE1.txt",
            "color": colors[2]
        },
        "Latin (modern)": {
            "wordvectors": "wordvectors/semeval/latin-corpus2.vec",
            "wordcounts": "results/word_counts/latin-LatinISE2.txt",
            "color": colors[2]
        },
        "Swedish (old)": {
            "wordvectors": "wordvectors/semeval/swedish-corpus1.vec",
            "wordcounts": "results/word_counts/swedish-old-kubhist2a.txt",
            "color": colors[3]
        },
        "Swedish (modern)": {
            "wordvectors": "wordvectors/semeval/swedish-corpus2.vec",
            "wordcounts": "results/word_counts/swedish-modern-kubhist2b.txt",
            "color": colors[3]
        },
        "UK Eng. (BNC)": {
            "wordvectors": "wordvectors/ukus/bnc.vec",
            "wordcounts": "results/word_counts/bnc.txt",
            "color": colors[4]
        },
        "US Eng. (COCA)": {
            "wordvectors": "wordvectors/ukus/coca.vec",
            "wordcounts": "results/word_counts/coca.txt",
            "color": colors[4]
        },
        "Spanish (old)": {
            "wordvectors": "wordvectors/spanish/old.vec",
            "wordcounts": "results/word_counts/dataset_XIX_lemmatized.txt",
            "color": colors[5]
        },
        "Spanish (modern)": {
            "wordvectors": "wordvectors/spanish/modern.vec",
            "wordcounts": "results/word_counts/modern_corpus_lemmatized.txt",
            "color": colors[5]
        }
    }

    # filter = ['NN']
    filter = None

    for d in datasets:
        print("---", d)

        wv = WordVectors(input_file=datasets[d]['wordvectors'])
        counts = load_word_counts(datasets[d]['wordcounts'])
        norms = get_vector_norms(wv)

        ws, cts, ns = get_distributions(counts, norms, filter=filter)

        print(len(ws), "words")

        sns.scatterplot(x=ns, y=cts, color=datasets[d]['color'])
        # sns.regplot(x=ns, y=cts, color="red")
        plt.xlabel("Word vector norm")
        plt.ylabel("Word frequency")
        plt.yscale("log")
        plt.title("%s [r=%.2f]" % (d, spearmanr(ns, cts).correlation))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, d+"-norms-counts.pdf"))
        plt.close()

