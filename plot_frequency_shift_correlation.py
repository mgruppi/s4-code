import pandas as pd
import os
import numpy as np
from scipy import stats
from scipy.stats import pearsonr
import re
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from direction_analysis import load_experiment_results
from frequency_norm_experiment import load_word_counts


if __name__ == "__main__":
    exp_path = 'results/direction/'
    files = sorted(os.listdir(exp_path))
    output_path = 'results/frequency-shift'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # df = pd.read_csv(os.path.join(exp_path, files[0]))

    data = {'k': list(), 'overlap': list(), 'dataset': list()}

    chosen_datasets = {
        # 'bnc-coca': 'UKUS English',
        # 'english-corpus1-english-corpus2': "English(SE2020)",
        # 'german-corpus1-german-corpus2': 'German',
        # 'latin-corpus1-latin-corpus2': 'Latin',
        # 'swedish-corpus1-swedish-corpus2': 'Swedish',
        # 'spanish-old-spanish-modern': 'Spanish',
        # 'wacad1990-wfic1990': 'Academic-Fiction',
        # 'wacad2000-wspok2000': 'Academic-Spoken',
        # 'wacad2012-wnews2012': 'Academic-News',
        # 'wacad2012-wfic2012': 'Academic-Fiction',
        # 'cs.AI-physics.class-ph': 'Ai-Physics (ArXiV)'
        'wacad1990-wacad2012': 'Academic 1990-2012',
        'wfic1990-wfic2012': 'Fiction 1990-2012',
        'wmag1990-wmag2012': 'Magazines 1990-2012',
        'wnews1990-wnews2012': 'News 1990-2012'
    }

    cmap = matplotlib.cm.get_cmap('Set2')
    colors = [cmap(i) for i in range(20)]

    dataset_freqs = {
        'bnc-coca': {
            'a': 'results/word_counts/bnc.txt',
            'b': 'results/word_counts/coca.txt',
            'color': colors[0]
        },
        'english-corpus1-english-corpus2': {
            'a': 'results/word_counts/english-corpus1.txt',
            'b': 'results/word_counts/english-corpus2.txt',
            'color': colors[1]
        },
        'german-corpus1-german-corpus2': {
            'a': 'results/word_counts/german-old-dta.txt',
            'b': 'results/word_counts/german-modern-bznd.txt',
            'color': colors[2]
        },
        'latin-corpus1-latin-corpus2': {
            'a': 'results/word_counts/latin-LatinISE1.txt',
            'b': 'results/word_counts/latin-LatinISE2.txt',
            'color': colors[3]
        },
        'swedish-corpus1-swedish-corpus2': {
            'a': 'results/word_counts/swedish-old-kubhist2a.txt',
            'b': 'results/word_counts/swedish-modern-kubhist2b.txt',
            'color': colors[4]
        },
        'spanish-old-spanish-modern': {
            'a': 'results/word_counts/dataset_XIX_lemmatized.txt',
            'b': 'results/word_counts/modern_corpus_lemmatized.txt',
            'color': colors[5]
        },
        'wacad1990-wfic1990': {
            'a': 'results/word_counts/coca/w_acad_1990.txt',
            'b': 'results/word_counts/coca/w_fic_1990.txt',
            'color': colors[0]
        },
        'wacad2000-wspok2000': {
            'a': 'results/word_counts/coca/w_acad_2000.txt',
            'b': 'results/word_counts/coca/w_spok_2000.txt',
            'color': colors[1]
        },
        'wacad2012-wnews2012': {
            'a': 'results/word_counts/coca/w_acad_2012.txt',
            'b': 'results/word_counts/coca/w_news_2012.txt',
            'color': colors[2]
        },
        'wacad2012-wfic2012': {
            'a': 'results/word_counts/coca/w_acad_2012.txt',
            'b': 'results/word_counts/coca/w_fic_2012.txt',
            'color': colors[3]
        },
        'wacad1990-wacad2012': {
            'a': 'results/word_counts/coca/w_acad_1990.txt',
            'b': 'results/word_counts/coca/w_acad_2012.txt',
            'color': colors[0]
        },
        'wfic1990-wfic2012': {
            'a': 'results/word_counts/coca/w_fic_1990.txt',
            'b': 'results/word_counts/coca/w_fic_2012.txt',
            'color': colors[1]
        },
        'wmag1990-wmag2012': {
            'a': 'results/word_counts/coca/w_mag_1990.txt',
            'b': 'results/word_counts/coca/w_mag_2012.txt',
            'color': colors[2]
        },
        'wnews1990-wnews2012': {
            'a': 'results/word_counts/coca/w_news_1990.txt',
            'b': 'results/word_counts/coca/w_news_2012.txt',
            'color': colors[4]
        },
    }

    # hue_order = ['English(SE2020)', 'German', 'Latin', 'Swedish', 'UKUS English', 'Spanish']
    hue_order = None
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 14})

    for i, f in enumerate(files):
        df, source_a, source_b = load_experiment_results(os.path.join(exp_path, f))
        ds = '%s-%s' % (source_a, source_b)

        print(" -", ds)
        df['freq'] = pd.Series([np.nan] * len(df))
        if ds in chosen_datasets:
            freq_a = load_word_counts(dataset_freqs[ds]['a'])
            freq_b = load_word_counts(dataset_freqs[ds]['b'])
            total_a = sum([v for v in freq_a.values()])
            total_b = sum([v for v in freq_b.values()])

            for row in df.itertuples():
                if row.direction == 'a_to_b':
                    if row.word in freq_a and row.word in freq_b:
                        # f = freq_b[row.word]/total_b - freq_a[row.word]/total_a
                        f = freq_a[row.word]+freq_b[row.word]
                        df.at[row.Index, 'freq'] = f
                else:
                    if row.word in freq_a and row.word in freq_b:
                        # f = freq_a[row.word]/total_a - freq_b[row.word]/total_b
                        f = freq_b[row.word]+freq_a[row.word]
                        df.at[row.Index, 'freq'] = f

            df = df.dropna()

            d = df[(df['direction'] == 'a_to_b') & (df['freq'] > 100)]
            df_sample = d.sample(n=500)
            pr = pearsonr(d['distance'], np.log(d['freq']))
            sns.scatterplot(data=df_sample, x='distance', y='freq', color=dataset_freqs[ds]['color'])
            plt.yscale('log')
            plt.title('r=%.2f' % pr[0])
            plt.ylabel('Frequency')
            plt.xlabel('Semantic Shift (cosine dist.)')
            plt.savefig(os.path.join(output_path, '%s.pdf' % ds))    
            plt.close()  
            print("---"*10)

    #     ds = '%s-%s' % (source_a, source_b)      
    #     print("---"*10)
    #     print(ds)
    #     x_k, x_kn, x_o = get_shift_rankings(df)
        
    #     if ds in chosen_datasets:
    #         data['k'].extend(x_k)
    #         data['overlap'].extend(x_o)
    #         data['dataset'].extend([chosen_datasets[ds]] * len(x_k))
    
    # plt.rcParams.update({'font.size': 14})
    # sns.set_style("whitegrid")
    # sns.lineplot(data=data, x='k', y='overlap', hue='dataset', hue_order=hue_order, palette='Accent')
    # plt.xlabel('k (depth)')
    # plt.ylabel('Overlap')
    # plt.savefig('results/direction-overlap.pdf')
    