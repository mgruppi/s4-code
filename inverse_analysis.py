import pandas as pd


if __name__ == "__main__":

    path_original = "results/predictions/predictions_param_search_semeval+ukus+spanish_normalized.csv"
    path_inverse = "results/predictions/predictions_param_search_semeval+ukus+spanish_normalized_inverse.csv"

    df_o = pd.read_csv(path_original)
    df_i = pd.read_csv(path_inverse)

    exp_params = {
        "English": {
            "dataset": "semeval_english",
            "cls": "cosine_050",
            "r": 3
        },
        "German": {
            "dataset": "semeval_german",
            "cls": "cosine_075",
            "r": 3
        },
        "Latin": {
            "dataset": "semeval_latin",
            "cls": "cosine_025",
            "r": 2
        },
        "Swedish": {
            "dataset": "semeval_swedish",
            "cls": "cosine_075",
            "r": 3
        }
    }

    for exp in exp_params:
        do = df_o[(df_o['dataset'] == exp_params[exp]['dataset']) \
                    & (df_o['cls'] == exp_params[exp]['cls']) \
                    & (df_o['r'].round() == exp_params[exp]['r'])]

        di = df_i[(df_i['dataset'] == exp_params[exp]['dataset']) \
                    & (df_i['cls'] == exp_params[exp]['cls']) \
                    & (df_i['r'].round() == exp_params[exp]['r'])]
        
        go = do.groupby(['word'])['correct'].mean().reset_index()
        gi = di.groupby(['word'])['correct'].mean().reset_index()

        print("===", exp)

        d = pd.merge(go, gi, on=['word'], how='inner', suffixes=['_o', '_inv'])

        print(d)

        