import pandas as pd
from WordVectors import WordVectors
import os
import numpy as np
from scipy import stats
import re
import matplotlib.pyplot as plt
import seaborn as sns


def load_experiment_results(path):
    """
    Loads the experiment CSV file and extract source names.

    Args:
        path(str) : Path to experiment CSV.
    
    Returns:
        df(pandas.DataFrame) : Results DataFrame.
        source_a(str) : Name of source a.
        source_b(str) : Name of source b.
    """

    df = pd.read_csv(path, on_bad_lines='skip', quoting=3)

    # After split: first is the word 'direction', second is source a, third source b.
    splits = os.path.basename(path).split('_')  # Split on '_'
    source_a = splits[1]
    source_b = splits[2]

    return df, source_a, source_b


def get_shift_rankings(df, k_lower=0.01, k_upper=1, k_step=51, print_rank_shift=False):
    """
    Returns the list of words and distances for each direction in input DataFrame `df`.
    Tables will be printed (in LaTeX format) for each dataset.

    Args:
        df(pandas.DataFrame) : DataFrame containing the experiment results.
        k_lower(float) : Lower bound for list depth in [0,1].
        k_upper(float) : Upper bound for list depth in [0,1].
        k_step(int) : Number of steps for k.
        print_rank_shift(bool) : If True, prints rank shift rather than pairwise rankings.

    Returns:
        x_k(list[float]) : List of list depth (fractions of common vocab.)
        x_k_n(list[int]) : List of depth size (as counts).
        x_o(list[float]) : List of overlap. Values within [0,1] where 0 is no overlap, 1 is complete overlap.
    """

    directions = sorted(df['direction'].unique())
    
    r_words = list()
    r_distances = list()

    df['word'] = df['word'].astype(str)
    df['alpha'] = pd.Series([False] * len(df))
    for row in df.itertuples():
        match = re.search('(\W+|\d)', row.word)  # Check if word is only alpha
        if match is None:
            df.at[row.Index, 'alpha'] = True
    
    df = df[df['alpha'] == True]

    for d in directions:
        df_ = df[df['direction'] == d]
        g = df_.groupby(['word'])['distance'].mean().reset_index()

        # if not print_rank_shift:
        #     df_sorted = g.sort_values('distance', ascending=False)  # Distance order
        # else:
        df_sorted = g.sort_values('word', ascending=True)  # Alphabetic order
        words = list(df_sorted['word'])
        distances = list(df_sorted['distance'])
        r_words.append(words)
        r_distances.append(distances)

    # Computes rank differences between directions
    a_rank = 1 + len(r_distances[0]) - stats.rankdata(r_distances[0], method='ordinal')  # rank high to low
    b_rank = 1 + len(r_distances[1]) - stats.rankdata(r_distances[1], method='ordinal')  # rank high to low
    rank_diff = a_rank - b_rank    
    rank_alpha = 0.01  # Alpha factor to apply to ranking priority
    rank_shift = np.fabs(rank_diff)/(np.exp(rank_alpha*a_rank) * np.exp(rank_alpha*b_rank))
    rank_shift_ord = np.argsort(rank_shift)[::-1]


    # Ranked order of word indices
    a_order = np.argsort(a_rank)
    b_order = np.argsort(b_rank)
    print("---"*10)
    n=15
    xsect = set.intersection(set(r_words[0][i] for i in a_order[:n]), set(r_words[1][i] for i in b_order[:n]))
    # Creates LaTeX table strings
    if not print_rank_shift:
        for i in range(n):
            a = r_words[0][a_order[i]]
            b = r_words[1][b_order[i]]
            if a == b:
                a = "\\cellcolor{matchColor}\\textbf{%s}" % a
                b = "\\cellcolor{matchColor}\\textbf{%s}" % b
            else:
                if a not in xsect:
                    a = "\\cellcolor{uniqueColor}\\textit{%s}" % a
                if b not in xsect:
                    b = "\\cellcolor{uniqueColor}\\textit{%s}" % b
                
            print("%s (%d-%d) & %s (%d-%d) \\\\" % (a, a_rank[a_order[i]], b_rank[a_order[i]], b, a_rank[b_order[i]], b_rank[b_order[i]]))
    else:
        for i in range(n):
            w_idx = rank_shift_ord[i]

            print("%s %d (%d-%d)" % (r_words[0][w_idx], rank_shift[w_idx], a_rank[w_idx], b_rank[w_idx]))

        # Use order of shift first
        # a_order = np.argsort(a_rank)
        # b_order = np.argsort(b_rank)
        # for i in range(n):
            # Get words with largest rank shift
            # w_idx = a_order[i]
            # w = r_words[0][w_idx]
            # w_rank_a = a_rank[w_idx]
            # w_rank_b = b_rank[w_idx]
            # print("%s %d (%d/%d)" % (w, w_rank_a-w_rank_b, w_rank_a, w_rank_b))
            # w_idx = rank_shift_ord[i]
            # w = r_words[0][w_idx]
            # print("%s (%d/%d)" % (w, a_rank[w_idx], b_rank[w_idx]))     
            
                
    # print("\n Kendall's Tau")
    # print(" - ", stats.kendalltau(r_words[0], r_words[1]))
    # print(" - (20)", stats.kendalltau(r_words[0][:20], r_words[1][:20]))
    # print("---")
    # print("---"*10)
    # k_list = np.arange(k_lower, k_upper+k_step, k_step)
    k_list = np.linspace(k_lower, k_upper, k_step)
    overlap_list = list()
    k_n_list = list()
    for k in k_list:
        k_n = int(k*len(r_words[0]))
        overlap = set.intersection(set(r_words[0][:k_n]), set(r_words[1][:k_n]))
        if k == k_lower or k == k_upper:
            print(k, k_n, len(overlap)/k_n)
        overlap_list.append(len(overlap)/k_n)
        k_n_list.append(k_n)
    
    k_list = np.array(k_list, dtype=float).round(decimals=1)
    return k_list, k_n_list, overlap_list


if __name__ == "__main__":
    exp_path = 'results/direction/'
    files = sorted(os.listdir(exp_path))


    # df = pd.read_csv(os.path.join(exp_path, files[0]))

    data = {'k': list(), 'overlap': list(), 'dataset': list()}

    chosen_datasets = {
        'bnc-coca': 'UKUS English',
        'english-corpus1-english-corpus2': "English(SE2020)",
        'german-corpus1-german-corpus2': 'German',
        'latin-corpus1-latin-corpus2': 'Latin',
        'swedish-corpus1-swedish-corpus2': 'Swedish',
        'spanish-old-spanish-modern': 'Spanish'
        # 'cs.AI-physics.class-ph': 'Ai-Physics (ArXiV)'
    }

    hue_order = ['English(SE2020)', 'German', 'Latin', 'Swedish', 'UKUS English', 'Spanish']

    for f in files:
        df, source_a, source_b = load_experiment_results(os.path.join(exp_path, f))

        ds = '%s-%s' % (source_a, source_b)      
        print("---"*10)
        print(ds)
        x_k, x_kn, x_o = get_shift_rankings(df, print_rank_shift=False)
        
        if ds in chosen_datasets:
            data['k'].extend(x_k)
            data['overlap'].extend(x_o)
            data['dataset'].extend([chosen_datasets[ds]] * len(x_k))
        
        input()
    
    plt.rcParams.update({'font.size': 14})
    sns.set_style("whitegrid")
    sns.lineplot(data=data, x='k', y='overlap', hue='dataset', hue_order=hue_order, palette='Accent')
    plt.xlabel('k (depth)')
    plt.ylabel('Overlap')
    plt.savefig('results/direction-overlap.pdf')
    
