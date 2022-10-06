import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    path = 'results/results_iters_search.csv'
    df = pd.read_csv(path)
    df = df[df['dataset'] == 'semeval_english']
    df = df[df['accuracy'] != -1]

    means = df.groupby(['iters', 'r']).mean().reset_index()
    print(means)

    sns.relplot(x='iters', y='accuracy', hue='r', data=means, kind='line')
    plt.ylim((0, 1))
    plt.savefig('means.png')

    # plt.plot(means.index, means['accuracy'])
    # plt.ylim((0,1))
    # plt.ylabel('accuracy')
    # plt.xlabel('Iter')
    # plt.savefig('means.png')