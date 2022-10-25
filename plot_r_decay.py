import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    pd.options.display.max_rows = 999  # Make Pandas display more rows during print
    plt.rcParams.update({'font.size': 14})
    sns.set_style("whitegrid")

    input_logs = 'results/r_decay/r_logs'
    input_scores = 'results/r_decay/scores'
    plot_dir = 'results/r_decay/plots'

    os.makedirs(plot_dir, exist_ok=True)  # exist_ok -> create dir only if it does not exist

    log_files = os.listdir(input_logs)
    score_files = os.listdir(input_scores)

    df_score_list = list()
    for f in score_files:
        df_f = pd.read_csv(os.path.join(input_scores, f))
        df_score_list.append(df_f)
    df_scores = pd.concat(df_score_list)
    g = df_scores.groupby(['dataset', 'r_max', 'r_min', 'r_decay'])['accuracy'].mean().round(decimals=2)
    print(g)

    df_list = list()
    for f in log_files:
        df_f = pd.read_csv(os.path.join(input_logs, f))
        dataset, rmax, rmin, rdecay = f.rstrip('.csv').rsplit("_", 3)
        rmax = float(rmax.replace('rmax', ''))
        rmin = float(rmin.replace('rmin', ''))
        rdecay = float(rdecay.replace('rdecay', ''))
        df_f['dataset'] = pd.Series([dataset] * len(df_f))
        df_f['r_max'] = pd.Series([rmax] * len(df_f))
        df_f['r_min'] = pd.Series([rmin] * len(df_f))
        df_f['r_decay'] = pd.Series([rdecay] * len(df_f))
        df_f[r"$\gamma$"] = df_f['r_decay']
        df_list.append(df_f)
    
    df = pd.concat(df_list)

    datasets = df['dataset'].unique()

    for ds in datasets:
        d = df[df['dataset'] == ds]
        # g = df.groupby(['r_max', 'r_min', 'r_decay'])['p_landmarks'].mean().reset_index()
        unique_rmax = d['r_max'].unique()

        # Weird workaround for "grouping" ('r_min' and 'r_max'), generating one plot for each combination
        # Warning: this is very slow!
        for rmax in unique_rmax:
            d_rm = d[d['r_max'] == rmax]
            unique_rmin = d_rm['r_min'].unique()
            
            for rmin in unique_rmin:
                d_ = d_rm[d_rm['r_min'] == rmin]

                fname = '%s_rmax%.2f_rmin%.2f.pdf' % (ds, rmax, rmin)
                sns.lineplot(data=d_, x='iter', y='p_landmarks', hue=r"$\gamma$", errorbar=None,
                             linewidth=2, palette='Set2')
                plt.tight_layout()
                plt.xlabel("Epoch")
                plt.ylabel("Landmarks")
                plt.savefig(os.path.join(plot_dir, fname))
                plt.close()

        # fname = '%s_rmax%.2f_rmin.pdf' % ()
    



        # df = pd.read_csv(os.path.join(input_logs, f))
        # fname = 'r_%s' % f.replace('.csv', '.pdf')
        # if 'p_landmarks' not in df.columns:
        #     continue
        # sns.lineplot(data=df, x='iter', y='p_landmarks', linewidth=2, errorbar=None)
        # plt.savefig(os.path.join(plot_dir, fname))
        # plt.close()
