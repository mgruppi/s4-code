"""
Runs parameter search for `r` in S4
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from WordVectors import WordVectors, intersection
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from s4 import s4, threshold_crossvalidation
from s4_torch import S4Network
from scipy.spatial.distance import cosine
from param_search_semeval import get_feature_cdf, vote
from alignment import align
import argparse


def cosine_cls(wv1, wv2, targets_1, targets_2, y_true, threshold=0.5, **kwargs):
    x = np.array([cosine(wv1[t1.lower()], wv2[t2.lower()]) for t1, t2 in zip(targets_1, targets_2)])
    y_pred = x.reshape(-1, 1)

    y_bin = (y_pred > threshold)
    accuracy = accuracy_score(y_true, y_bin)
    precision = precision_score(y_true, y_bin)
    recall = recall_score(y_true, y_bin)
    f1 = f1_score(y_true, y_bin)

    return accuracy, precision, recall, f1


def s4_cls(wv1, wv2, targets_1, targets_2, y_true, **kwargs):
    cls_model = S4Network(wv1.dimension*2)
    model = s4(wv1, wv2, update_landmarks=False,
               verbose=0,
               cls_model=cls_model,
               **kwargs)
    x = np.array([np.concatenate((wv1[t1.lower()], wv2[t2.lower()])) for t1, t2 in zip(targets_1, targets_2)])
    y_pred = model.predict(x)
    y_bin = y_pred > 0.5

    accuracy = accuracy_score(y_true, y_bin)
    precision = precision_score(y_true, y_bin)
    recall = recall_score(y_true, y_bin)
    f1 = f1_score(y_true, y_bin)

    return accuracy, precision, recall, f1


def run_experiments(wv1, wv2, targets_1, targets_2, y_true, num_trials=10, r_upper=2,
                    cls=cosine_cls, r_steps=11,
                    align_method="s4a",
                    **kwargs):
    """
    Performs experiments by varying R in a range for a given input
    Args:
        wv1: (WordVectors) input WordVectors 1
        wv2: (WordVectors) input WordVectors 2
        targets_1: (array-like) Target words in wv1
        targets_2: (array-like) Target words in wv2
        y_true: (array-like) True labels of the target words
        num_trials: (int) Number of times to repeat the trial for each `r`
        r_upper: (int) Upper bound for parameter `r`
        cls: (callable) Classifier to apply. Must receive wv1, wv2, targets and y_true as parameters
        r_steps: (int) Number of steps in which to increase `r`
        align_method: (str) Alignment strategy to apply in {'s4a', 'global'}

    Returns:
        results: Returns a list of tuples with the results (r, accuracy, precision, recall, f1)
    """
    np.random.seed(1)
    r_range = np.linspace(0, r_upper, r_steps)
    print("R range", r_range)

    results = list()

    cls_model = S4Network(wv1.dimension*2)

    for r_ in r_range:
        for i in range(num_trials):
            if align_method == "global":
                landmarks = wv1.words
            elif align_method == "s4a":
                landmarks, non_landmarks, Q, = s4(wv1, wv2,
                                                  verbose=0,
                                                  rate=r_,
                                                  cls_model=cls_model,
                                                  iters=20
                                                  )
            wv1_, wv2_, Q = align(wv1, wv2, anchor_words=landmarks)
            acc, prec, rec, f1 = cls(wv1_, wv2_, targets_1, targets_2, y_true, landmarks=landmarks, **kwargs)
            res_tuple = (r_, acc, prec, rec, f1)
            print(*res_tuple, sep=",")
            results.append(res_tuple)
    return results


def n_experiment_generator(wv1, wv2, targets_1, targets_2, y_true, num_trials=10,
                           cls=cosine_cls,
                           r=1,
                           n_steps=250,
                           n_pos_upper=1500,
                           n_neg_upper=1500,
                           align_method="s4a",
                           **kwargs):
    """
    Performs experiments by varying R in a range for a given input
    Args:
        wv1: (WordVectors) input WordVectors 1
        wv2: (WordVectors) input WordVectors 2
        targets_1: (array-like) Target words in wv1
        targets_2: (array-like) Target words in wv2
        y_true: (array-like) True labels of the target words
        num_trials: (int) Number of times to repeat the trial for each `r`
        cls: (callable) Classifier to apply. Must receive wv1, wv2, targets and y_true as parameters
        r: (float) R parameter to use
        n_steps: (int) Size of steps to take
        n_pos_upper: (int) Upper bound for n positives
        n_neg_upper: (int) Upper bound for n negatives
        align_method: (str) Alignment strategy to apply in {'s4a', 'global'}

    Returns:
        result_tuple: Yields tuples with the results (r, n_pos, n_neg, accuracy, precision, recall, f1)
    """
    np.random.seed(1)
    n_pos_range = np.arange(100, n_pos_upper+100, n_steps)
    print("N_pos range", n_pos_range)
    n_neg_range = np.arange(100, n_neg_upper+100, n_steps)
    print("N_neg range", n_neg_range)

    cls_model = S4Network(wv1.dimension*2)

    for n_pos in n_pos_range:
        for n_neg in n_neg_range:
            for i in range(num_trials):
                if align_method == "global":
                    landmarks = wv1.words
                elif align_method == "s4a":
                    landmarks, non_landmarks, Q, = s4(wv1, wv2,
                                                      verbose=0,
                                                      rate=r,
                                                      n_targets=n_pos,
                                                      n_negatives=n_neg,
                                                      cls_model=cls_model,
                                                      iters=20,
                                                      )
                wv1_, wv2_, Q = align(wv1, wv2, anchor_words=landmarks)
                acc, prec, rec, f1 = cls(wv1_, wv2_, targets_1, targets_2, y_true, landmarks=landmarks, **kwargs)
                res_tuple = (r, n_pos, n_neg, acc, prec, rec, f1)
                print(*res_tuple, sep=",")
                yield res_tuple


def read_semeval_data(lang, normalized=False):
    # Load SemEval 2020 data
    corpus1_path = "wordvectors/semeval/%s-corpus1.vec" % lang
    corpus2_path = "wordvectors/semeval/%s-corpus2.vec" % lang
    wv1 = WordVectors(input_file=corpus1_path, normalized=normalized)
    wv2 = WordVectors(input_file=corpus2_path, normalized=normalized)
    wv1, wv2 = intersection(wv1, wv2)

    path_task1 = "data/semeval/truth/%s.txt" % lang
    with open(path_task1) as fin:
        data = map(lambda s: s.strip().split("\t"), fin.readlines())
        targets, true_class = zip(*data)
        y_true = np.array(true_class, dtype=int)

    return wv1, wv2, targets, y_true


def read_ukus_data(normalized=False):
    path_us = "wordvectors/ukus/coca.vec"
    path_uk = "wordvectors/ukus/bnc.vec"
    path_dict = "data/ukus/dict_similar.txt"
    path_dict_dis = "data/ukus/dict_dissimilar.txt"

    wv1 = WordVectors(input_file=path_uk, normalized=normalized)
    wv2 = WordVectors(input_file=path_us, normalized=normalized)
    wv_uk, wv_us = intersection(wv1, wv2)

    # Load dictionaries of words
    with open(path_dict) as fin:
        dico_sim = list(map(lambda s: s.strip().split(" ", 1), fin.readlines()))

    with open(path_dict_dis) as fin:
        dico_dis = list(map(lambda s: (s.strip(), s.strip()), fin.readlines()))

    # Filter words not in the vocabulry of either UK or US corpora
    dico_sim = [(a, b) for a, b in dico_sim if a in wv_uk.word_id and b in wv_us.word_id]
    dico_dis = [(a, b) for a, b in dico_dis if a in wv_uk.word_id and b in wv_us.word_id]
    dico = dico_sim + dico_dis

    # Create true labels for terms
    # 0 -> similar | 1 -> dissimilar
    y_true = [0] * len(dico_sim) + [1]*len(dico_dis)

    return wv_uk, wv_us, dico, y_true


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("param", type=str, choices=["r", "n"],
                        help="Type of parameter to search. r - rate of perturbation | n - no. of samples to perturb")
    parser.add_argument("--no-semeval", dest="no_semeval", action="store_true",
                        help="Do not perform SemEval 2020 experiment")
    parser.add_argument("--no-ukus", dest="no_ukus", action="store_true",
                        help="Do not perform UKUS experiment")
    parser.add_argument("--num-trials", dest="num_trials", type=int, default=10,
                        help="Number of trials per r value")
    parser.add_argument("--normalized", action="store_true", help="Normalize word vectors")
    parser.add_argument("--iters", type=int, default=100, help="No. of s4 iterations (epochs)")

    args = parser.parse_args()

    normalized = args.normalized
    languages = ["english", "german", "latin", "swedish"]

    if args.param == "r":
        if not args.no_semeval:
            semeval_params = [{"threshold": 0.01, "cls": cosine_cls}, {"threshold": 0.25, "cls": cosine_cls},
                              {"threshold": 0.5, "cls": cosine_cls}, {"threshold": 0.75, "cls": cosine_cls},
                              {"threshold": 0.9, "cls": cosine_cls}]
            cls_names = ["cosine_001", "cosine_025", "cosine_050", "cosine_075",
                         "cosine_090"]

            fout = open("param_search_results_semeval.txt", "w")
            fout.write("language,cls_name,r,accuracy,precision,recall,f1\n")

            for lang in languages:
                wv1, wv2, targets, y_true = read_semeval_data(lang, normalized)

                for name, params in zip(cls_names, semeval_params):
                    results_semeval = run_experiments(wv1, wv2, targets, targets, y_true,
                                                      num_trials=args.num_trials,
                                                      **params
                                                      )
                    for res in results_semeval:
                        print(lang, name, *res, sep=",", file=fout)

            fout.close()

        if not args.no_ukus:
            cls_names = ["cosine_03", "cosine_05", "cosine_07", "s4d"]
            ukus_params = [{"threshold": 0.3, "cls": cosine_cls}, {"threshold": 0.5, "cls": cosine_cls},
                           {"threshold": 0.7, "cls": cosine_cls}, {"cls": s4_cls}]

            fout = open("param_search_results_ukus.txt", "w")
            fout.write("cls_name,r,accuracy,precision,recall,f1\n")
            wv1, wv2, targets, y_true = read_ukus_data(normalized)
            targets_1, targets_2 = zip(*targets)
            for name, params in zip(cls_names, ukus_params):
                results_ukus = run_experiments(wv1, wv2, targets_1, targets_2, y_true,
                                               num_trials=args.num_trials,
                                               align_method="global",
                                               **params)

                for res in results_ukus:
                    print(name, *res, sep=",", file=fout)

            fout.close()
    elif args.param == "n":
        r_value = 1.0
        if not args.no_semeval:
            semeval_params = [{"threshold": 0.01, "cls": cosine_cls},
                              {"threshold": 0.25, "cls": cosine_cls},
                              {"threshold": 0.5, "cls": cosine_cls},
                              {"threshold": 0.75, "cls": cosine_cls},
                              {"threshold": 0.9, "cls": cosine_cls}]
            cls_names = ["cosine_001", "cosine_025", "cosine_050", "cosine_075",
                         "cosine_090"]
            fout = open("param_search_n_results_semeval.txt", "w")
            fout.write("language,cls_name,r,n_pos,n_neg,accuracy,precision,recall,f1\n")
            for lang in languages:
                wv1, wv2, targets, y_true = read_semeval_data(lang, normalized)

                for name, params in zip(cls_names, semeval_params):
                    # results_semeval = run_experiments(wv1, wv2, targets, targets, y_true,
                    #                                   num_trials=args.num_trials,
                    #                                   **params
                    #                                   )
                    n_experiments = n_experiment_generator(wv1, wv2, targets, targets, y_true,
                                                           num_trials=args.num_trials,
                                                           **params)
                    for res in n_experiments:
                        print(lang, name, *res, sep=",", file=fout)
            fout.close()

        if not args.no_ukus:
            cls_names = ["cosine_03", "cosine_05", "cosine_07", "s4d"]
            ukus_params = [{"threshold": 0.3, "cls": cosine_cls}, {"threshold": 0.5, "cls": cosine_cls},
                           {"threshold": 0.7, "cls": cosine_cls}, {"cls": s4_cls}]

            fout = open("param_search_n_results_ukus.txt", "w")
            fout.write("cls_name,r,n_pos,n_neg,accuracy,precision,recall,f1\n")
            wv1, wv2, targets, y_true = read_ukus_data(normalized)
            targets_1, targets_2 = zip(*targets)
            for name, params in zip(cls_names, ukus_params):
                n_experiments = n_experiment_generator(wv1, wv2, targets_1, targets_2, y_true,
                                                       num_trials=args.num_trials,
                                                       align_method="global",
                                                       **params)

                for res in n_experiments:
                    print(name, *res, sep=",", file=fout)

