import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str,
                    help="Path to CSV file containing the experiment results.")
parser.add_argument("--parameter", choices=["n", "r", "choice"], default="n", help="Parameter to plot (r or n)")
parser.add_argument("--output", type=str, default=None,
                    help="Path to save plots to")

args = parser.parse_args()
df = pd.read_csv(args.input)

if args.output is None:
    output = "results/"

if not os.path.exists(output):
    os.makedirs(output)

metrics = ["accuracy", "precision", "recall", "f1"]
classifiers = ["cosine_050", "cosine_025", "cosine_075"]


# Drop invalid entries (-1 scores)
df = df.drop(df[df['accuracy'] == -1].index)  # drop invalid rows 

# Compute positive_accuracy and negative_accuracy
positive_accuracy = df['true_positives']/(df['true_positives'] + df['false_negatives'])
negative_accuracy = df['true_negatives']/(df['true_negatives'] + df['false_positives'])
df['positive_accuracy'] = positive_accuracy
df['negative_accuracy'] = negative_accuracy

datasets = list(df['dataset'].unique())

# Plot parameter `r`
path_out = os.path.join(output, "r")
if not os.path.exists(path_out):
    os.makedirs(path_out)

for ds in datasets:
    df_d = df[df['dataset'] == ds]

    
    for m in metrics:
        sns.relplot(data=df_d, x="r", y=m, kind="line", hue="cls")
        plt.savefig(os.path.join(path_out, "%s_%s.png" % (ds, m)))
        plt.close()
    
    # Plot positive/negative accuracies
    dfm = df_d[['r','cls','positive_accuracy','negative_accuracy']].melt(['r', 'cls'], var_name='group', value_name='vals')

    sns.relplot(data=dfm, x="r", y="vals", kind="line", hue="cls", style='group', ci=None)

    plt.savefig(os.path.join(path_out, "%s_posneg.png" % ds))
        




