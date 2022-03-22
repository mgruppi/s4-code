import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os


f_se = "param_search_results_semeval.txt"
f_en = "param_search_results_ukus.txt"

path_out = "results/r_search/"
if not os.path.exists(path_out):
    os.makedirs(path_out)

df_se = pd.read_csv(f_se)

print(df_se)
lang = "english"
cls = "cosine_010"
metric = "accuracy"

languages = ["english", "german", "latin", "swedish"]
metrics = ["accuracy", "precision", "recall", "f1"]

for lang in languages:
    df = df_se[df_se["language"] == lang]

    for m in metrics:
        sns.relplot(data=df, x="r", y=m, kind="line", hue="cls_name")
        plt.savefig(os.path.join(path_out, "semeval_%s_%s.png" % (lang, m)))


df_en = pd.read_csv(f_en)
for m in metrics:
    sns.relplot(data=df_en, x="r", y=m, kind="line", hue="cls_name")
    plt.savefig(os.path.join(path_out, "ukus_%s.png" % m))

# files = ["param_search_results_semeval_cosine.txt", "param_search_results_semeval_s4.txt"]
#
# for f in files:
#     df = pd.read_csv(f, index_col=False)
#     sns.relplot(data=df, x="r", y="accuracy", kind="line", hue="language")
#     plt.legend()
#     plt.show()
#
#
# files_ukus = ["param_search_results_ukus_cosine_03.txt", "param_search_results_ukus_s4.txt"]
#
# for f in files:
#     df = pd.read_csv(f, index_col=False)
#     sns.relplot(data=df, x="r", y="accuracy", kind="line")
#     plt.legend()
#     plt.show()


# df = pd.read_csv("results_param_search_semeval.txt", sep="|", header=0, index_col=False,
#                  names=["language", "mean_accuracy", "max_accuracy", "r"])
#
# print(df)
#
# for lang in df["language"].unique():
#     print(lang)
#     x = df[df["language"] == lang]
#     plt.plot(x["r"], x["mean_accuracy"], label=lang)
#
#
# plt.xlabel("r")
# plt.ylabel("acc")
# plt.legend()
# plt.show()
#
#
# df_u = pd.read_csv("results_r_ukus.txt", sep="|", header=0, index_col=False,
#                    names=["cls_method", "align_method", "acc", "prec", "recall", "f1", "r"])
#
# for cls in df_u["cls_method"].unique():
#     print(cls)
#     x = df_u[df_u["cls_method"] == cls]
#     plt.plot(x["r"], x["acc"], label=cls)
#
# plt.xlabel("r")
# plt.ylabel("acc")
# plt.legend()
# plt.show()
