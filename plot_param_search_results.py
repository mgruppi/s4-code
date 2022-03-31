import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os


f_se = "param_search_n_results_semeval.txt"
f_en = "param_search_n_results_ukus.txt"

path_out_r = "results/r_search/"
path_out_n = "results/n_search/"
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
parameter = "n"

# Parameter r plot
if parameter == "r":  # skip
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
        for m in metrics:
            sns.relplot(data=df, x="n_pos", y=m, kind="line", hue="cls_name")
            plt.tight_layout()
            plt.savefig(os.path.join(path_out_n, "semeval_n_pos_%s_%s.png") % (lang, m))
            plt.close()
            sns.relplot(data=df, x="n_neg", y=m, kind="line", hue="cls_name")
            plt.tight_layout()
            plt.savefig(os.path.join(path_out_n, "semeval_n_neg_%s_%s.png") % (lang, m))
            plt.close()
            sns.relplot(data=df, x="n_diff", y=m, kind="line", hue="cls_name")
            plt.tight_layout()
            plt.savefig(os.path.join(path_out_n, "semeval_n_diff_%s_%s.png" % (lang, m)))
            plt.close()

    df_en["n_diff"] = (df_en["n_pos"] - df_en["n_neg"]) / max(df_en["n_pos"])
    for m in metrics:
        sns.relplot(data=df_en, x="n_pos", y=m, kind="line", hue="cls_name")
        plt.savefig(os.path.join(path_out_n, "ukus_%s.png" % m))
        plt.close()
        sns.relplot(data=df_en, x="n_neg", y=m, kind="line", hue="cls_name")
        plt.savefig(os.path.join(path_out_n, "ukus_%s.png" % m))
        plt.close()
        sns.relplot(data=df_en, x="n_diff", y=m, kind="line", hue="cls_name")
        plt.savefig(os.path.join(path_out_n, "ukus_%s.png" % m))
        plt.close()

