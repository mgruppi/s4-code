import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

files = ["param_search_results_semeval_cosine.txt", "param_search_results_semeval_s4.txt"]

for f in files:
    df = pd.read_csv(f, index_col=False)
    sns.relplot(data=df, x="r", y="accuracy", kind="line", hue="language")
    plt.legend()
    plt.show()


files_ukus = ["param_search_results_ukus_cosine_03.txt", "param_search_results_ukus_s4.txt"]

for f in files:
    df = pd.read_csv(f, index_col=False)
    sns.relplot(data=df, x="r", y="accuracy", kind="line")
    plt.legend()
    plt.show()


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
