import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import pandas as pd


if __name__ == "__main__":
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 14})
    input_file = "results/normalized/param_search_r_results_semeval.txt"
    input_file_norm = "results/normalized/param_search_r_results_semeval_normalized.txt"

    input_file_uk = "results/normalized/param_search_r_results_ukus.txt"
    input_file_ukus = "results/normalized/param_search_r_results_ukus_normalized.txt"

    df_0 = pd.read_csv(input_file)
    df_n = pd.read_csv(input_file_norm)

    # Plot SemEval-2020
    g = df_0.groupby(["language", "cls_name"]).mean()
    gn = df_n.groupby(["language", "cls_name"]).mean()

    g = g.groupby(['language'])['accuracy'].max().reset_index()
    g['Norm'] = pd.Series(['Non-normalized']*len(g))

    gn = gn.groupby(['language'])['accuracy'].max().reset_index()
    gn['Norm'] = pd.Series(['Normalized']*len(gn))

    d = pd.concat((g, gn))

    sns.barplot(x='language', y='accuracy', data=d, hue='Norm', palette='Accent')

    plt.ylabel("Accuracy")
    plt.xlabel("Dataset")

    plt.ylim(0,1)

    plt.tight_layout()
    plt.savefig("figures/normalized_comparison.pdf")

    # results = {'dataset': list(), 'cls': list(), 'accuracy': list()}

    # for k in g:
    #     results['dataset'].append(k[0])
    #     results['cls'].append(k[1])
    #     results['accuracy'].append(g[k])
    

