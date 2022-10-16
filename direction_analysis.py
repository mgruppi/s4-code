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


def get_shift_rankings(df, k_lower=0.01, k_upper=1, k_step=51):
    """
    Returns the list of words and distances for each direction in input DataFrame `df`.
    Tables will be printed (in LaTeX format) for each dataset.

    Args:
        df(pandas.DataFrame) : DataFrame containing the experiment results.
        k_lower(float) : Lower bound for list depth in [0,1].
        k_upper(float) : Upper bound for list depth in [0,1].
        k_step(int) : Number of steps for k.

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
        df_sorted = g.sort_values('distance', ascending=False)
        words = list(df_sorted['word'])
        distances = list(df_sorted['distance'])
        r_words.append(words)
        r_distances.append(distances)

    print("---"*10)
    n=15
    # Creates LaTeX table strings
    for i in range(n):
        xsect = set.intersection(set(r_words[0][:n]), set(r_words[1][:n]))
        a = r_words[0][i]
        b = r_words[1][i]
        if r_words[0][i] == r_words[1][i]:
            a = "\\textbf{%s}" % a
            b = "\\textbf{%s}" % b
        else:
            if a not in xsect:
                a = "\\textit{%s}" % a
            if b not in xsect:
                b = "\\textit{%s}" % b
            
        print("%s & %s \\\\" % (a, b))
    
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
        x_k, x_kn, x_o = get_shift_rankings(df)
        
        if ds in chosen_datasets:
            data['k'].extend(x_k)
            data['overlap'].extend(x_o)
            data['dataset'].extend([chosen_datasets[ds]] * len(x_k))
    
    plt.rcParams.update({'font.size': 14})
    sns.set_style("whitegrid")
    sns.lineplot(data=data, x='k', y='overlap', hue='dataset', hue_order=hue_order, palette='Accent')
    plt.xlabel('k (depth)')
    plt.ylabel('Overlap')
    plt.savefig('results/direction-overlap.pdf')
    
