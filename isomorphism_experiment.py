from WordVectors import WordVectors
from alignment import align
import os
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def run_alignment_experiments(wv_base, wv_list):
    """
    Runs the alignment experiments.

    Args:
        wv_base(WordVectors) : Base word vectors (to align to).
        wv_list(list[WordVectors]) : Word vectors to align and compute distances.
        choice(str) : How to choose landmarks. Must be in {'top', 'bottom', 'random'}, resulting in choosing
        the top most frequent words (top), least frequent words (bottom), or random.

    Returns:
        df(pd.DataFrame) : DataFrame containing the experiment results.
    """

    # k_range = np.linspace(1, len(wv_base.words), 5, dtype=int)  # Align in 10 steps
    k_range = [1, 50, 100, 200, 500, 1000, len(wv_base.words)]
    # k_range = np.arange(1, len(wv_base.words), 5)
    data = list()
    header = ('k', 'word', 'distance', 'choice')
    choices = ['top', 'bottom', 'random']

    for choice in choices:
        for k in k_range:
            if k < 1:
                n_landmarks = int(k*len(wv_base.words))
            else:
                n_landmarks = k
            print(' - k =', k, '(%d)' % n_landmarks)
            if choice == 'top':
                landmarks = wv_base.words[:n_landmarks]
            elif choice == 'bottom':
                landmarks = wv_base.words[-n_landmarks:]
            elif choice == 'random':
                landmarks = np.random.choice(wv_base.words, size=n_landmarks)
            
            wv1, wv2, Q = align(wv_list[0], wv_base, anchor_words=landmarks)

            for word in wv1.words:
                data.append((k, word, cosine(wv1[word], wv2[word]), choice))
    
    df = pd.DataFrame(data, columns=header)
    return df


if __name__ == "__main__":
    """
    Performs an experiment to investigate whether word embeddings trained on the same corpus with different initializations
    can be aligned to each other in a meaningful way. That is, that multiple word embeddings are isomorphic (maintaining its inner distances).
    """
    
    input_path = 'wordvectors/isomorphism'
    output_path = 'results/isomorphism'
    files = os.listdir(input_path)

    np.random.seed(1)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    wv_base = WordVectors(input_file=os.path.join(input_path, files[0]), normalized=True)

    wv_list = list()
    for f in files[1:2]:
        wv = WordVectors(input_file=os.path.join(input_path, f))
        wv_list.append(wv)

    df = run_alignment_experiments(wv_base, wv_list)

    # sns.histplot(data=df, x='distance', hue='k')
    sns.set_style("whitegrid")
    sns.lineplot(data=df, x='k', y='distance', hue='choice', estimator='mean', errorbar='se', palette='Accent')
    plt.xlabel("Landmarks")
    plt.ylabel("Cosine distance")
    plt.xscale('log')
    plt.savefig(os.path.join(output_path, 'cosine_dist.pdf'))
    plt.close()

    df = df[df['k'] == 1000]
    sns.histplot(data=df, x='distance', palette='Accent')
    plt.xlabel('Cosine distance')
    plt.savefig(os.path.join(output_path, 'cosine_histogram.pdf'))

    print("Done")
    

