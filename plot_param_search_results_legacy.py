import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--f_se", type=str, default="ablations/param_search_n_results_semeval.txt",
                    help="Path to semeval experiment data.")
parser.add_argument("--f_en", type=str, default="ablations/param_search_n_results_ukus.txt",
                    help="Path to UKUS experiment data.")
parser.add_argument("--parameter", choices=["n", "r", "choice"], default="n", help="Parameter to plot (r or n)")

args = parser.parse_args()

f_se = args.f_se
f_en = args.f_en

path_out_r = "results/r_search/"
path_out_n = "results/n_search/"
path_out_choice = "results/choice_method/"
if not os.path.exists(path_out_r):
    os.makedirs(path_out_r)

df_se = pd.read_csv(f_se)
df_en = pd.read_csv(f_en)

print(df_se)
lang = "english"
cls = "cosine_010"
metric = "accuracy"

languages = ["english", "german", "latin", "swedish"]
metrics = ["accuracy", "precision", "recall", "f1"]
classifiers = ["cosine_050", "cosine_025", "cosine_075"]
parameter = args.parameter

# Parameter r plot
if parameter == "r":
    for lang in languages:
        n_pos = 100
        n_neg = 100
        df = df_se[(df_se["language"] == lang) & (df_se["n_pos"] == n_pos) & (df_se["n_neg"] == n_neg)]

        for m in metrics:
            sns.relplot(data=df, x="r", y=m, kind="line", hue="cls_name")
            plt.savefig(os.path.join(path_out_r, "semeval_%s_%s.png" % (lang, m)))

    for m in metrics:
        sns.relplot(data=df_en, x="r", y=m, kind="line", hue="cls_name")
        plt.savefig(os.path.join(path_out_r, "ukus_%s.png" % m))

# Parameter n plot

elif parameter == "n":
    for lang in languages:
        r = 1
        df_se["n_diff"] = (df_se["n_pos"]-df_se["n_neg"])/max(df_se["n_pos"])
        df = df_se[(df_se["language"] == lang) & (df_se["r"] == r)]
        unique_n_pos = sorted(df["n_pos"].unique())
        unique_n_neg = sorted(df["n_neg"].unique())[::-1]
        # We call reset_index() to convert the MultiIndex back into columns
        # df = df.groupby(["n_pos", "n_neg", "cls_name"]).mean().reset_index()
        df = df.groupby(["n_pos", "n_neg", "cls_name"]).mean()

        for cls_name in classifiers:
            for m in metrics:
                # v_min and v_max are used to determine the range of the color scale and fix it for each language
                v_min = df[m].min()  
                v_max = df[m].max()
                x = np.zeros((len(unique_n_pos), len(unique_n_neg)))
                mask = np.zeros(x.shape)
                for i, n_p in enumerate(unique_n_pos):
                    for j, n_n in enumerate(unique_n_neg):
                        # print(n_p, n_n, cls_name)
                        # print(df.loc[(n_p, n_n, cls_name)])
                        try:
                            x[i][j] = df.loc[(n_p, n_n, cls_name)][m]
                        except KeyError as e:
                            mask[i][j] = 1
                            print("Index not found - ", n_p, n_n, cls_name, " - ", e)

                sns.heatmap(x, mask=mask, xticklabels=unique_n_pos, yticklabels=unique_n_neg,
                            annot=True, vmin=v_min, vmax=v_max)
                plt.title("%s - %s - %s" % (lang, m, cls_name))
                plt.xlabel("n_pos")
                plt.ylabel("n_neg")
                plt.tight_layout()
                output_dir = os.path.join(path_out_n, "%s" % cls_name)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                plt.savefig(os.path.join(output_dir, "semeval_heatmap_%s_%s.png") % (lang, m))
                plt.close()
            # sns.relplot(data=df, x="n_pos", y=m, kind="line", hue="cls_name")
            # plt.tight_layout()
            # plt.savefig(os.path.join(path_out_n, "semeval_n_pos_%s_%s.png") % (lang, m))
            # plt.close()
            # sns.relplot(data=df, x="n_neg", y=m, kind="line", hue="cls_name")
            # plt.tight_layout()
            # plt.savefig(os.path.join(path_out_n, "semeval_n_neg_%s_%s.png") % (lang, m))
            # plt.close()
            # sns.relplot(data=df, x="n_diff", y=m, kind="line", hue="cls_name")
            # plt.tight_layout()
            # plt.savefig(os.path.join(path_out_n, "semeval_n_diff_%s_%s.png" % (lang, m)))

    unique_n_pos = sorted(df_en["n_pos"].unique())
    unique_n_neg = sorted(df_en["n_neg"].unique())[::-1]
    df = df_en.groupby(["n_pos", "n_neg", "cls_name"]).mean()
    cls_name = "s4d"
    for m in metrics:
        x = np.zeros((len(unique_n_pos), len(unique_n_neg)))
        for i, n_p in enumerate(unique_n_pos):
            for j, n_n in enumerate(unique_n_neg):
                # print(n_p, n_n, cls_name)
                # print(df.loc[(n_p, n_n, cls_name)])
                try:
                    x[i][j] = df.loc[(n_p, n_n, cls_name)][m]
                except KeyError as e:
                    print("Index not found - ", n_p, n_n, cls_name, " - ", e)
        sns.heatmap(x, xticklabels=unique_n_pos, yticklabels=unique_n_neg)
        plt.title("UK-US - %s" % m)
        plt.xlabel("n_pos")
        plt.ylabel("n_neg")
        plt.tight_layout()
        plt.savefig(os.path.join(path_out_n, "ukus_heatmap_%s.png") % m)
        plt.close()

elif parameter == 'choice':
    # SemEval
    for lang in languages:
        n_pos = 500
        n_neg = 500
        df = df_se[(df_se["language"] == lang) & (df_se["n_pos"] == n_pos) & (df_se["n_neg"] == n_neg)]
        
        for cls_name in classifiers:
            for m in metrics:
                df_ = df[df["cls_name"] == cls_name]
                sns.boxplot(data=df_, x="choice_method", y=m)
                output_dir = os.path.join(path_out_choice, "%s" % cls_name)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                plt.savefig(os.path.join(output_dir, "semeval_%s_%s.png" % (lang, m)))
                plt.close()

    # UK-US
    for m in metrics:
        sns.boxplot(data=df, x="choice_method", y=m)
        plt.savefig(os.path.join(path_out_choice, "ukus_%s.png" % m))
        plt.close()

