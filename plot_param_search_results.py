import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns
import os
import numpy as np
import argparse
from scipy import stats
import json


parser = argparse.ArgumentParser()
parser.add_argument("input", type=str,
                    help="Path to CSV file containing the experiment results.")
parser.add_argument("--parameter", choices=["n", "r", "choice"], default="r", help="Parameter to plot (r or n)")
parser.add_argument("--output", type=str, default=None,
                    help="Path to save plots to")

matplotlib.rcParams.update({"font.size": 14})
sns.set_style("whitegrid")

cmap = matplotlib.cm.get_cmap('Accent')
colors = [cmap(i) for i in range(10)]

args = parser.parse_args()
df = pd.read_csv(args.input)

if args.output is None:
    output = "results/"
else:
    output = args.output

with open('data/metadata.json') as fin:
    metadata = json.load(fin)

if not os.path.exists(output):
    os.makedirs(output)

metrics = ["accuracy", "precision", "recall", "f1", "landmarks", "landmarks_norm"]
classifiers = ["cosine_050", "cosine_025", "cosine_075"]

dataset_names = {
    "semeval_english": "English (SE2020)",
    "semeval_german": "German",
    "semeval_latin": "Latin",
    "semeval_swedish": "Swedish",
    "ukus": "UK-US English",
    "spanish": "Spanish"
}

# Drop invalid entries (-1 scores)
df = df.drop(df[df['accuracy'] == -1].index)  # drop invalid rows 

df = df[df['cls'].isin(classifiers)]
df['r'] = df['r'].astype(float).round(decimals=2)
df['landmarks_norm'] = pd.Series([0] * len(df), dtype=float)
df['n_pos_norm'] = pd.Series([0] * len(df), dtype=float)
df['n_neg_norm'] = pd.Series([0]* len(df), dtype=float)

# Normalize landmarks
for row in df.itertuples():
    df.at[row.Index, 'landmarks_norm'] = row.landmarks/metadata[row.dataset]['common_vocab_size']
    df.at[row.Index, 'n_pos_norm'] = row.n_pos/metadata[row.dataset]['common_vocab_size']
    df.at[row.Index, 'n_neg_norm'] = row.n_neg/metadata[row.dataset]['common_vocab_size']

df['n_pos_norm'] = (df['n_pos_norm']*100).astype(int)
df['n_neg_norm'] = (df['n_neg_norm']*100).astype(int)
df['landmarks_norm'] = df['landmarks_norm'].round(decimals=2)

# df['landmarks_norm'] = pd.Series([l/metadata[v]['common_vocab_size']
#                                     for l, v in zip(df['landmarks'], df['dataset'])])

# Compute positive_accuracy and negative_accuracy
positive_accuracy = df['true_positives']/(df['true_positives'] + df['false_negatives'])
negative_accuracy = df['true_negatives']/(df['true_negatives'] + df['false_positives'])
df['positive_accuracy'] = positive_accuracy
df['negative_accuracy'] = negative_accuracy

datasets = list(df['dataset'].unique())


if args.parameter == 'r':
    # Plot parameter `r`
    path_out = os.path.join(output, "r")
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    # Plot landmarks distribution
    axes = sns.lineplot(data=df, x='r', y='landmarks_norm', hue='dataset', palette='Accent', \
                        hue_order=datasets, linewidth=2, legend=False)
    for ds, line in zip(datasets, axes.get_lines()):
        yoffset = 0
        if ds == 'ukus':
            yoffset=-0.004
        c = line.get_color()
        if ds == 'semeval_swedish':
            c = 'gold'
        axes.annotate(dataset_names[ds], (line.get_xdata()[-1], line.get_ydata()[-1]+yoffset), \
                        color=c)
    plt.xlim(0,4)
    plt.ylabel("Landmarks")
    plt.tight_layout()
    plt.savefig(os.path.join(path_out, 'landmarks_dist.pdf'))
    plt.close()

    for ds in datasets:
        df_d = df[df['dataset'] == ds]
        
        for m in metrics:
            if m == 'landmarks' or m == 'landmarks_norm':
                hue = None
            else:
                hue = 'cls'

            # if m == 'landmarks_norm':
            #     ln = df_d['landmarks']/metadata[ds]['common_vocab_size']
            #     df_d['landmarks_norm'] = ln

            sns.lineplot(data=df_d, x="r", y=m, hue=hue, errorbar='se', palette='Set2', linewidth=3)
            plt.ylabel(m.capitalize())
            plt.tight_layout()
            plt.savefig(os.path.join(path_out, "%s_%s.pdf" % (ds, m)))
            plt.close()
        
        # # Plot positive/negative accuracies
        # dfm = df_d[['r','cls','positive_accuracy','negative_accuracy']].melt(['r', 'cls'], var_name='group', value_name='vals')
        # sns.relplot(data=dfm, x="r", y="vals", kind="line", hue="cls", style='group', errorbar=None)
        # plt.savefig(os.path.join(path_out, "%s_posneg.pdf" % ds))
        # plt.close()
elif args.parameter == 'n':
    path_out = os.path.join(output, 'n')
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    num_p = df['n_pos_norm'].nunique()
    num_n = df['n_neg_norm'].nunique()
    x_final = np.zeros((num_p, num_n), dtype=float)
    c_final = 0

    for ds in datasets:
        df_d = df[df['dataset'] == ds]

        unique_n_pos_norm = sorted(df_d['n_pos_norm'].unique())
        unique_n_neg_norm = sorted(df_d['n_neg_norm'].unique())[::-1]
        classifiers = df_d['cls'].unique()


        df_d = df_d.groupby(['n_pos_norm', 'n_neg_norm', 'cls']).mean()
        for cls_name in classifiers:
            for m in metrics:
                v_min = df_d[m].min()
                v_max = df_d[m].max()
                x = np.zeros((len(unique_n_pos_norm), len(unique_n_neg_norm)))
                mask = np.zeros_like(x)

                for i, n_p in enumerate(unique_n_pos_norm):
                    for j, n_n in enumerate(unique_n_neg_norm):
                        try:
                            x[i][j] = df_d.loc[n_p, n_n, cls_name][m]
                        except KeyError as e:
                            x[i][j] = v_min
                            print("Index not found - ", n_p, n_n, cls_name, ds, " - ", e)

                sns.heatmap(x, mask=mask, xticklabels=unique_n_pos_norm, yticklabels=unique_n_neg_norm,
                            annot=True, vmin=v_min, vmax=v_max, 
                            # cbar_kws={'label': m.capitalize()},
                            cmap='magma')
                # plt.title('%s - %s -%s' % (ds, m, cls_name))
                plt.xlabel('Positive samples (%)')
                plt.ylabel('Negative samples (%)')
                plt.tight_layout()
                output_dir = os.path.join(path_out, cls_name)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                plt.savefig(os.path.join(output_dir, '%s_heatmap_%s.pdf' % (ds, m)))
                plt.close()
            if m == 'landmarks_norm':
                x_final += x
                c_final += 1
    x_final = x_final/c_final
    ax = sns.heatmap(x_final, xticklabels=unique_n_pos_norm, yticklabels=unique_n_neg_norm, 
                annot=False, cmap='YlGnBu', cbar_kws={'label': 'Landmarks'})
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    plt.xlabel('Positive samples (%)')
    plt.ylabel('Negative samples (%)')
    plt.savefig(os.path.join(path_out, 'landmarks_heatmap.pdf'))
    plt.close()

        

