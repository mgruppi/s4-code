import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    np.random.seed(1)
    plt.rcParams.update({'font.size': 14})

    x = [[0.01, 0.02]]

    x_mid = np.random.normal((0.2, 0.1), 0.1, size=(20, 2))
    x_far = np.random.normal((0.4, 0.4), 0.1, size=(8,2))

    sns.set_style("whitegrid")

    fig, ax = plt.subplots()

    ax.scatter(*x[0], color='dodgerblue')
    ax.annotate(r"$s_w^A$", x[0], (0.022, 0.025))
    c1 = plt.Circle((x[0]), 0.012, fill=False, color='dodgerblue')
    c2 = plt.Circle((0.4, 0.4), 0.18, fill=True, edgecolor='black', clip_on=True, facecolor='orange', alpha=.2)

    ax.annotate(r"Candidate sentences", (0.25, 0.4))

    for p in x_mid:
        ax.scatter(*p, color='gray')
    for p in x_far:
        ax.scatter(*p, color='orange')
    
    cfar = plt.Circle((x_far[0]), 0.012, fill=False, color='orange')
    ax.annotate(r"$s_w^B$", (x_far[0][0]+0.01, x_far[0][1]-0.02))

    ax.set_xlim(-0.01, 0.5)
    ax.set_ylim(-0.01, 0.5)

    ax.add_patch(c1)
    ax.add_patch(c2)
    ax.add_patch(cfar)

    fig.savefig('sentence_candidates.pdf')