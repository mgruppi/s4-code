from WordVectors import WordVectors, intersection
from alignment import align
import os
from s4 import s4
from parameter_search import read_semeval_data, read_ukus_data, read_spanish_data, cosine_cls
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


if __name__ == "__main__":

    output_dir = 'results/r_decay'
    n_trials = 3
    r_max = 5
    r_min = 1
    # r_decay = 0.95
    max_iters=100
    np = 200
    nn = 200

    r_decay_list = [1, 0.99, 0.5, 0.25]

    result_header = ('dataset', 'iter', 'r_max', 'r_min', 'r_decay', 'iters', 'n_pos', 'n_neg', 'choice_method', 'alignment', 'cls', \
                 'landmarks', 'p_landmarks', 'accuracy', 'precision', 'recall', 'f1', 'true_negatives', 'false_positives', \
                 'false_negatives', 'true_positives')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(os.path.join(output_dir, 'r_logs')):
        os.makedirs(os.path.join(output_dir, 'r_logs'))
    if not os.path.exists(os.path.join(output_dir, 'scores')):
        os.makedirs(os.path.join(output_dir, 'scores'))

    normalized=True
    languages = ['english', 'german', 'latin', 'swedish']

    best_cls = {
        "english": {
            "cls_names": ["cosine_050"],
            "cls_func": [cosine_cls],
            "cls_thresholds": [0.5]
        },
        "latin": {
            "cls_names": ["cosine_025"],
            "cls_func": [cosine_cls],
            "cls_thresholds": [0.25]
        },
        "german": {
            "cls_names": ["cosine_075"],
            "cls_func": [cosine_cls],
            "cls_thresholds": [0.75]
        },
        "swedish": {
            "cls_names": ["cosine_075"],
            "cls_func": [cosine_cls],
            "cls_thresholds": [0.75]
        },
        "spanish": {
            "cls_names": ["cosine_075"],
            "cls_func": [cosine_cls],
            "cls_thresholds": [0.75]
        },
        "ukus": {
            "cls_names": ["cosine_025"],
            "cls_func": [cosine_cls],
            "cls_thresholds": [0.25]
        }
    }

    for r_decay in r_decay_list:

        for lang in languages:
            df_list = list()
            result_list = list()

            cls = best_cls[lang]['cls_func'][0]
            thrs = best_cls[lang]['cls_thresholds'][0]
            cls_name = best_cls[lang]['cls_names'][0]

            dataset = 'semeval_%s' % lang

            for i in range(n_trials):
                wv1, wv2, targets, y_true = read_semeval_data(lang, normalized)  
                targets_1 = targets_2 = targets

                landmarks, non_landmarks, Q, df_history = s4(wv1, wv2,
                                                                verbose=1,
                                                                rate=r_max,
                                                                r_decay=r_decay,
                                                                rate_min=r_min,
                                                                log_history=True,
                                                                n_targets=200,
                                                                n_negatives=200,
                                                                cls_model='nn',
                                                                iters=max_iters
                                                            )
                df_list.append(df_history)
                wv1, wv2, Q = align(wv1, wv2, anchor_words=landmarks)

                # Score predictions
                n_landmarks = len(landmarks)
                p_landmarks = n_landmarks/len(wv1.words)  # Normalized landmarks
                acc, prec, rec, f1, tn, fp, fn, tp, correct, incorrect = cls(wv1, wv2, targets_1, targets_2, y_true, \
                                                                                landmarks=landmarks, threshold=thrs)
                res_tuple = (dataset, i, r_max, r_min, r_decay, max_iters, np, nn, 'random', 's4a', cls_name, n_landmarks, p_landmarks, acc, prec, rec, f1, tn, fp, fn, tp)
                result_list.append(res_tuple)
            
            # Dump scores
            df_result = pd.DataFrame(result_list, columns=result_header)
            result_file = os.path.join(output_dir, 'scores', '%s_rmax%.2f_rmin%.2f_rdecay%.2f_results.csv' % (dataset, r_max, r_min, r_decay))
            df_result.to_csv(result_file, index=False)

            # Dump history
            df = pd.concat(df_list)
            log_file = os.path.join(output_dir, 'r_logs', '%s_rmax%.2f_rmin%.2f_rdecay%.2f.csv' % (dataset, r_max, r_min, r_decay))
            img_file = log_file.replace('.csv', '.pdf')
            df.to_csv(log_file, index=False)
        
        # # === UK-US
        # dataset = 'ukus'
        # cls = best_cls[dataset]['cls_func'][0]
        # thrs = best_cls[dataset]['cls_thresholds'][0]
        # cls_name = best_cls[dataset]['cls_names'][0]

        # df_list = list()
        # result_list = list()

        # for i in range(n_trials):
        #     wv1, wv2, targets, y_true = read_ukus_data(normalized)
        #     targets_1, targets_2 = zip(*targets)
        #     landmarks, non_landmarks, Q, df_history = s4(wv1, wv2,
        #                                                     verbose=1,
        #                                                     rate=r_max,
        #                                                     r_decay=r_decay,
        #                                                     rate_min=r_min,
        #                                                     log_history=True,
        #                                                     n_targets=200,
        #                                                     n_negatives=200,
        #                                                     cls_model='nn',
        #                                                     iters=max_iters
        #                                                 )
        #     df_list.append(df_history)

        #     # Score predictions
        #     n_landmarks = len(landmarks)
        #     p_landmarks = n_landmarks/len(wv1.words)  # Normalized landmarks
        #     acc, prec, rec, f1, tn, fp, fn, tp, correct, incorrect = cls(wv1, wv2, targets_1, targets_2, y_true, \
        #                                                                     landmarks=landmarks, threshold=thrs)
        #     res_tuple = (dataset, i, r_max, r_min, r_decay, max_iters, np, nn, 'random', 's4a', cls_name, n_landmarks, p_landmarks, acc, prec, rec, f1, tn, fp, fn, tp)
        #     result_list.append(res_tuple)
        
        # # Dump scores
        # df_result = pd.DataFrame(result_list, columns=result_header)
        # result_file = os.path.join(output_dir, 'scores', '%s_rmax%.2f_rmin%.2f_rdecay%.2f_results.csv' % (dataset, r_max, r_min, r_decay))
        # df_result.to_csv(result_file, index=False)

        # df = pd.concat(df_list)
        # log_file = os.path.join(output_dir, 'r_logs', '%s_rmax%.2f_rmin%.2f_rdecay%.2f.csv' % (dataset, r_max, r_min, r_decay))
        # img_file = log_file.replace('.csv', '.pdf')
        # df.to_csv(log_file, index=False)

        # # === Spanish
        # cls = best_cls[dataset]['cls_func'][0]
        # thrs = best_cls[dataset]['cls_thresholds'][0]
        # cls_name = best_cls[dataset]['cls_names'][0]

        # df_list = list()
        # result_list = list()

        # for i in range(n_trials):
        #     wv1, wv2, targets, y_true = read_spanish_data(normalized)
        #     targets_1 = targets_2 = targets
        #     landmarks, non_landmarks, Q, df_history = s4(wv1, wv2,
        #                                                     verbose=1,
        #                                                     rate=r_max,
        #                                                     r_decay=r_decay,
        #                                                     rate_min=r_min,
        #                                                     log_history=True,
        #                                                     n_targets=200,
        #                                                     n_negatives=200,
        #                                                     cls_model='nn',
        #                                                     iters=max_iters
        #                                                 )
        #     df_list.append(df_history)
        
        #     # Score predictions
        #     n_landmarks = len(landmarks)
        #     p_landmarks = n_landmarks/len(wv1.words)  # Normalized landmarks
        #     acc, prec, rec, f1, tn, fp, fn, tp, correct, incorrect = cls(wv1, wv2, targets_1, targets_2, y_true, \
        #                                                                     landmarks=landmarks, threshold=thrs)
        #     res_tuple = (dataset, i, r_max, r_min, r_decay, max_iters, np, nn, 'random', 's4a', cls_name, n_landmarks, p_landmarks, acc, prec, rec, f1, tn, fp, fn, tp)
        #     result_list.append(res_tuple)
        
        # # Dump scores
        # df_result = pd.DataFrame(result_list, columns=result_header)
        # result_file = os.path.join(output_dir, 'scores', '%s_rmax%.2f_rmin%.2f_rdecay%.2f_results.csv' % (dataset, r_max, r_min, r_decay))
        # df_result.to_csv(result_file, index=False)
        
        # df = pd.concat(df_list)
        # dataset = 'spanish'
        # log_file = os.path.join(output_dir,'r_logs', '%s_rmax%.2f_rmin%.2f_rdecay%.2f.csv' % (dataset, r_max, r_min, r_decay))
        # img_file = log_file.replace('.csv', '.pdf')
        # df.to_csv(log_file, index=False)


