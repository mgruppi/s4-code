"""
Runs parameter search for `r` in S4
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
from WordVectors import WordVectors, intersection
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from s4 import s4, threshold_crossvalidation
from s4_torch import S4Network
from scipy.spatial.distance import cosine
from param_search_semeval import get_feature_cdf, vote
from alignment import align
import argparse
import itertools


def cosine_cls(wv1, wv2, targets_1, targets_2, y_true, threshold=0.5, **kwargs):
    x = np.array([cosine(wv1[t1.lower()], wv2[t2.lower()]) for t1, t2 in zip(targets_1, targets_2)])
    y_pred = x.reshape(-1, 1)

    y_bin = (y_pred > threshold)
    accuracy = accuracy_score(y_true, y_bin)
    precision = precision_score(y_true, y_bin)
    recall = recall_score(y_true, y_bin)
    f1 = f1_score(y_true, y_bin)
    tn, fp, fn, tp = confusion_matrix(y_true, y_bin).ravel()

    return accuracy, precision, recall, f1, tn, fp, fn, tp


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
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return accuracy, precision, recall, f1, tn, fp, fn, tp
     

def choice_experiments(wv1, wv2, targets_1, targets_2, y_true, num_trials=10, r=1.0,
                    cls=cosine_cls,
                    align_method="s4a",
                    n_pos=500,
                    n_neg=500,
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
        r: (float) Parameter `r`
        cls: (callable) Classifier to apply. Must receive wv1, wv2, targets and y_true as parameters
        align_method: (str) Alignment strategy to apply in {'s4a', 'global'}

    Returns:
        results: Returns a list of tuples with the results (r, accuracy, precision, recall, f1)
    """
    np.random.seed(1)

    choice_methods = ['random', 'far', 'close']

    for c in choice_methods:
        cls_model = S4Network(wv1.dimension*2)
        for i in range(num_trials):
            if align_method == "global":
                landmarks = wv1.words
            elif align_method == "s4a":
                landmarks, non_landmarks, Q, = s4(wv1, wv2,
                                                    verbose=0,
                                                    rate=r,
                                                    cls_model=cls_model,
                                                    n_targets=n_pos,
                                                    n_negatives=n_neg,
                                                    iters=20,
                                                    inject_choice=c
                                                    )
            n_landmarks = len(landmarks)
            if n_landmarks > 0:
                wv1_, wv2_, Q = align(wv1, wv2, anchor_words=landmarks)
                acc, prec, rec, f1 = cls(wv1_, wv2_, targets_1, targets_2, y_true, landmarks=landmarks, **kwargs)
                res_tuple = (c, r, n_pos, n_neg, n_landmarks, acc, prec, rec, f1)
                print(*res_tuple, sep=",")
            else:
                res_tuple = ()
            yield res_tuple


def run_r_experiments(wv1, wv2, targets_1, targets_2, y_true, num_trials=10, r_upper=2,
                    cls=cosine_cls, r_steps=11,
                    align_method="s4a",
                    n_pos=100,
                    n_neg=100,
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
                                                  n_targets=n_pos,
                                                  n_negatives=n_neg,
                                                  iters=20
                                                  )
            # wv1_, wv2_, Q = align(wv1, wv2, anchor_words=landmarks)
            # acc, prec, rec, f1 = cls(wv1_, wv2_, targets_1, targets_2, y_true, landmarks=landmarks, **kwargs)
            # res_tuple = (r_, acc, prec, rec, f1)
            # print(*res_tuple, sep=",")
            # yield res_tuple

            n_landmarks = len(landmarks)
            if n_landmarks > 0:
                wv1_, wv2_, Q = align(wv1, wv2, anchor_words=landmarks)
                acc, prec, rec, f1 = cls(wv1_, wv2_, targets_1, targets_2, y_true, landmarks=landmarks, **kwargs)
                res_tuple = (r_, n_pos, n_neg, n_landmarks, acc, prec, rec, f1)
                print(*res_tuple, sep=",")
            else:
                res_tuple = ()
            yield res_tuple


def n_experiment_generator(wv1, wv2, targets_1, targets_2, y_true, num_trials=10,
                           cls=cosine_cls,
                           r=1,
                           n_steps=500,
                           n_pos_upper=5000,
                           n_neg_upper=5000,
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
    n_pos_range = np.arange(200, n_pos_upper+100, n_steps)
    print("N_pos range", n_pos_range)
    n_neg_range = np.arange(200, n_neg_upper+100, n_steps)
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
                n_landmarks = len(landmarks)
                if n_landmarks > 0:
                    wv1_, wv2_, Q = align(wv1, wv2, anchor_words=landmarks)
                    acc, prec, rec, f1 = cls(wv1_, wv2_, targets_1, targets_2, y_true, landmarks=landmarks, **kwargs)
                    res_tuple = (r, n_pos, n_neg, n_landmarks, acc, prec, rec, f1)
                    print(*res_tuple, sep=",")
                else:
                    res_tuple = ()
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


def read_spanish_data(normalized=False, truth_column="change_binary"):
    """
    Reads spanish word vectors + ground truth data from the LChange2022 Shared Task.
    Arguments:
        normalized: Boolean value for reading normalized WordVectors
        truth_column: Name of the truth column to use. Defaults to "change_binary".
                        Other options are "change_binary_gain" and "change_binary_loss".
    """
    path_old = "wordvectors/spanish/old.vec"
    path_modern = "wordvectors/spanish/modern.vec"

    wv1 = WordVectors(input_file=path_old, normalized=normalized)
    wv2 = WordVectors(input_file=path_modern, normalized=normalized)
    wv_old, wv_mod = intersection(wv1, wv2)

    df = pd.read_csv("data/spanish/stats_groupings.csv", sep="\t")
    targets = df["lemma"]
    y_true = df[truth_column]

    return wv_old, wv_mod, targets, y_true


def old_main():

    parser = argparse.ArgumentParser()
    parser.add_argument("param", type=str, choices=["r", "n", "choice_method"],
                        help="Type of parameter to search. r - rate of perturbation | n - no. of samples to perturb")
    parser.add_argument("--no-semeval", dest="no_semeval", action="store_true",
                        help="Do not perform SemEval 2020 experiment")
    parser.add_argument("--no-ukus", dest="no_ukus", action="store_true",
                        help="Do not perform UKUS experiment")
    parser.add_argument("--no-spanish", dest="no_spanish", action="store_true",
                        help="Do not perform Spanish experiment")
    parser.add_argument("--num-trials", dest="num_trials", type=int, default=10,
                        help="Number of trials per r value")
    parser.add_argument("--normalized", action="store_true", help="Normalize word vectors")
    parser.add_argument("--languages", default=None, nargs="+", help="List of languages")
    parser.add_argument("--r-upper", dest="r_upper", default=2, type=float, help="Upper bound for r")
    parser.add_argument("--r-steps", dest="r_steps", default=11, type=int, help="No. of steps for r")

    args = parser.parse_args()

    normalized = args.normalized

    if args.languages is None:
        languages = ["english", "german", "latin", "swedish"]
    else:
        languages = args.languages

    if args.param == "r":
        if not args.no_semeval:
            semeval_params = [{"threshold": 0.01, "cls": cosine_cls}, {"threshold": 0.25, "cls": cosine_cls},
                              {"threshold": 0.5, "cls": cosine_cls}, {"threshold": 0.75, "cls": cosine_cls},
                              {"threshold": 0.9, "cls": cosine_cls}]
            cls_names = ["cosine_001", "cosine_025", "cosine_050", "cosine_075",
                         "cosine_090"]

            fout = open("param_search_r_results_semeval.txt", "w")
            fout.write("language,cls_name,r,n_pos,n_neg,n_landmarks,accuracy,precision,recall,f1\n")

            for lang in languages:
                wv1, wv2, targets, y_true = read_semeval_data(lang, normalized)

                for name, params in zip(cls_names, semeval_params):
                    results_semeval = run_r_experiments(wv1, wv2, targets, targets, y_true,
                                                      num_trials=args.num_trials,
                                                      r_upper=args.r_upper,
                                                      r_steps=args.r_steps,
                                                      **params
                                                      )
                    for res in results_semeval:
                        print(lang, name, *res, sep=",", file=fout)

            fout.close()

        if not args.no_ukus:
            cls_names = ["cosine_03", "cosine_05", "cosine_07", "s4d"]
            ukus_params = [{"threshold": 0.3, "cls": cosine_cls}, {"threshold": 0.5, "cls": cosine_cls},
                           {"threshold": 0.7, "cls": cosine_cls}, {"cls": s4_cls}]

            fout = open("param_search_r_results_ukus.txt", "w")
            fout.write("cls_name,r,n_pos,n_neg,n_landmarks,accuracy,precision,recall,f1\n")
            wv1, wv2, targets, y_true = read_ukus_data(normalized)
            targets_1, targets_2 = zip(*targets)
            for name, params in zip(cls_names, ukus_params):
                results_ukus = run_r_experiments(wv1, wv2, targets_1, targets_2, y_true,
                                               num_trials=args.num_trials,
                                               r_upper=args.r_upper,
                                               r_steps=args.r_steps,
                                               align_method="global",
                                               **params)

                for res in results_ukus:
                    print(name, *res, sep=",", file=fout)

            fout.close()
        if not args.no_spanish:
            cls_names = ["cosine_025"]
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
            fout.write("language,cls_name,r,n_pos,n_neg,n_landmarks,accuracy,precision,recall,f1\n")
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
                        if len(res) > 0:
                            print(lang, name, *res, sep=",", file=fout)
            fout.close()

        if not args.no_ukus:
            cls_names = ["cosine_03", "cosine_05", "cosine_07", "s4d"]
            ukus_params = [{"threshold": 0.3, "cls": cosine_cls}, {"threshold": 0.5, "cls": cosine_cls},
                           {"threshold": 0.7, "cls": cosine_cls}, {"cls": s4_cls}]

            fout = open("param_search_n_results_ukus.txt", "w")
            fout.write("cls_name,r,n_pos,n_neg,n_landmarks,accuracy,precision,recall,f1\n")
            wv1, wv2, targets, y_true = read_ukus_data(normalized)
            targets_1, targets_2 = zip(*targets)
            for name, params in zip(cls_names, ukus_params):
                n_experiments = n_experiment_generator(wv1, wv2, targets_1, targets_2, y_true,
                                                       num_trials=args.num_trials,
                                                       align_method="global",
                                                       **params)

                for res in n_experiments:
                    if len(res) > 0:
                        print(name, *res, sep=",", file=fout)

    elif args.param == 'choice_method':
        if not args.no_semeval:
            semeval_params = [{"threshold": 0.01, "cls": cosine_cls},
                              {"threshold": 0.25, "cls": cosine_cls},
                              {"threshold": 0.5, "cls": cosine_cls},
                              {"threshold": 0.75, "cls": cosine_cls},
                              {"threshold": 0.9, "cls": cosine_cls}]
            cls_names = ["cosine_001", "cosine_025", "cosine_050", "cosine_075",
                         "cosine_090"]
            fout = open("param_search_choice_method_results_semeval.txt", "w")
            fout.write("language,cls_name,choice_method,r,n_pos,n_neg,n_landmarks,accuracy,precision,recall,f1\n")
            for lang in languages:
                wv1, wv2, targets, y_true = read_semeval_data(lang, normalized)

                for name, params in zip(cls_names, semeval_params):
                    # results_semeval = run_experiments(wv1, wv2, targets, targets, y_true,
                    #                                   num_trials=args.num_trials,
                    #                                   **params
                    #                                   )
                    n_experiments = choice_experiments(wv1, wv2, targets, targets, y_true,
                                                           num_trials=args.num_trials,
                                                           **params)
                    for res in n_experiments:
                        if len(res) > 0:
                            print(lang, name, *res, sep=",", file=fout)
            fout.close()

        if not args.no_ukus:
            cls_names = ["cosine_03", "cosine_05", "cosine_07", "s4d"]
            ukus_params = [{"threshold": 0.3, "cls": cosine_cls}, {"threshold": 0.5, "cls": cosine_cls},
                           {"threshold": 0.7, "cls": cosine_cls}, {"cls": s4_cls}]

            fout = open("param_search_choice_method_results_ukus.txt", "w")
            fout.write("cls_name,choice_method,r,n_pos,n_neg,n_landmarks,accuracy,precision,recall,f1\n")
            wv1, wv2, targets, y_true = read_ukus_data(normalized)
            targets_1, targets_2 = zip(*targets)
            for name, params in zip(cls_names, ukus_params):
                n_experiments = choice_experiments(wv1, wv2, targets_1, targets_2, y_true,
                                                       num_trials=args.num_trials,
                                                       align_method="global",
                                                       **params)

                for res in n_experiments:
                    if len(res) > 0:
                        print(name, *res, sep=",", file=fout)


def run_experiments(wv1, wv2, targets_1, targets_2, y_true,
                    r, n_pos, n_neg, choice_method, align_method, classifier,
                    num_trials=10,
                    **kwargs):
    """
    Performs experiments with a given setup. The number of settings is determined by which parameters are variable.
    Particularly, each of the `r`, `n_pos`, `n_neg`, `choice_method`, `align_method`, and `cls` can be varied, resulting in up to
    `s * t * u * v * w * x`, parameter combinations. 
    Typically, several parameters would be fixed and only one or few varied.
    Args:
        wv1:    (WordVectors)
        wv2:    (WordVectors)
        targets_1:  (iterable) Target words in wv1  
        targets_2:  (iterable) Target words in wv2
        y_true: (iterable) True label of the target words
        r:      (float or iterable) If `float`, the fixed value of parameter `r`. If iterable, the list of values of `r` to apply.
        n_pos:  (float or iterable) If `float`, the fixed number of positive samples to generate. If `iterable`, the list of possible `n_pos` values.
        n_neg:  (float or iterable) If `float`, the fixed number of negative samples to generate. If `iterable`, the list of possible `n_neg` values.
        choice_method:  (string or iterable) If `string`, the fixed perturbation method to apply. If `iterable`, the list of perturbations method to use.
        align_method:   (string or iterable) If `string`, the fixed alignment method to use. If `iterable`, the list of alignment methods to use
        classifier:  (tuple or iterable) If `tuple`, then `(name, func, threshold)` must be passed as the name, callable and threshold of the classifier. If `iterable`, the list of classifier names)
        num_trials: (int) The number of trials to run for each setting.
        **kwargs:   Keyword arguments to send to S4.
    Returns:
        (header, results): Generator for the experiments. `header` is a tuple containing the name of the returned fields,
        `results` is a tuple containing the value of the returned fields.
    """
    np.random.seed(1)
    if not hasattr(r, '__iter__'):
        r = [r]  # Turn fixed r in to a list
    if not hasattr(n_pos, '__iter__'):
        n_pos = [n_pos]
    if not hasattr(n_neg, '__iter__'):
        n_neg = [n_neg]
    if type(choice_method) == str or not hasattr(choice_method, '__iter__'):
        choice_method = [choice_method]
    if type(align_method) == str or not hasattr(align_method, '__iter__'):
        align_method = [align_method]
    if not hasattr(classifier, '__iter__'):
        classifier = [classifier]
    
    exp_settings = itertools.product(r, n_pos, n_neg, choice_method, align_method, classifier)
    for _r, _np, _nn, _cm, _am, _cls in exp_settings:
        print(_r, _np, _nn, _cm, _am, _cls)

        for i in range(num_trials):
            if _am == "global":
                landmarks = wv1.words
            elif _am == "s4a":
                cls_model = S4Network(wv1.dimension*2)  # This should not be the classifier from `_cls`, this is the internal S4 model
                landmarks, non_landmarks, Q, =  s4(wv1, wv2,
                                                    verbose=0,
                                                    rate=_r,
                                                    n_targets=_np,
                                                    n_negatives=_nn,
                                                    cls_model=cls_model,
                                                    iters=20, 
                                                )
            n_landmarks = len(landmarks)
            header_tuple = ('r', 'n_pos', 'n_neg', 'choice_method', 'alignment', 'cls', 'accuracy', 'precision',
                'recall', 'f1', 'true_negatives', 'false_positives', 'false_negatives', 'true_positives')
            if n_landmarks > 0:
                wv1_, wv2_, Q = align(wv1, wv2, anchor_words=landmarks)
                acc, prec, rec, f1, tn, fp, fn, tp = _cls[1](wv1_, wv2_, targets_1, targets_2, y_true, landmarks=landmarks, threshold=_cls[2])
                res_tuple = (_r, _np, _nn, _cm, _am, _cls[0], acc, prec, rec, f1, tn, fp, fn, tp)
            else:
                res_tuple = (_r, _np, _nn, _cm, _am, _cls[0], -1, -1, -1, -1, -1, -1, -1, -1)
            yield (header_tuple, res_tuple)


def new_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("param", type=str, choices=["r", "n", "choice_method", "all"],
                        help="Type of parameter to search. r - rate of perturbation | n - no. of samples to perturb")
    parser.add_argument("--no-semeval", dest="no_semeval", action="store_true",
                        help="Do not perform SemEval 2020 experiment")
    parser.add_argument("--no-ukus", dest="no_ukus", action="store_true",
                        help="Do not perform UKUS experiment")
    parser.add_argument("--no-spanish", dest="no_spanish", action="store_true",
                        help="Do not perform Spanish experiment")
    parser.add_argument("--num-trials", dest="num_trials", type=int, default=10,
                        help="Number of trials per r value")
    parser.add_argument("--normalized", action="store_true", help="Normalize word vectors")
    parser.add_argument("--languages", default=None, nargs="+", help="List of languages")
    parser.add_argument("--r-upper", dest="r_upper", default=2, type=float, help="Upper bound for r")
    parser.add_argument("--r-steps", dest="r_steps", default=11, type=int, help="No. of steps for r")
    parser.add_argument("--output-file", dest='output_file', default='results_param_search.csv', type=str,
                        help='Change default output file')

    args = parser.parse_args()

    normalized = args.normalized

    if args.languages is None:
        languages = ["english", "german", "latin", "swedish"]
    else:
        languages = args.languages

    # Make file header
    header_tuple = ('dataset', 'normalized', 'r', 'n_pos', 'n_neg', 'choice_method', 'alignment', 'cls', 'accuracy', 'precision',
    'recall', 'f1', 'true_negatives', 'false_positives', 'false_negatives', 'true_positives')
    fout = open(args.output_file, 'w')
    print(*header_tuple, sep=',', file=fout)

    if args.param == "all":  # all vs all comparison
        n_pos_range = np.arange(500, 5500, 500)
        print("N_pos range", n_pos_range)
        n_neg_range = np.arange(500, 5500, 500)
        print("N_neg range", n_neg_range)

        r_range = np.linspace(0, 5, 10)
        print("R-range", r_range)

        choice_methods = ['random', 'far', 'close']
        print("Choice methods", choice_methods)

        align_methods = ['s4a']

        cls_names = ['cosine_025', 'cosine_050', 'cosine_075', 'cosine_090']
        cls_func = [cosine_cls, cosine_cls, cosine_cls, cosine_cls]
        cls_thresholds = [0.25, 0.5, 0.75, 0.90]
        classifiers = list(zip(cls_names, cls_func, cls_thresholds))
    elif args.param == "r":
        n_pos_range = [500]
        n_neg_range = [500]
        choice_methods = ['random']
        align_methods = ['s4a']
        cls_names = ['cosine_025', 'cosine_050', 'cosine_075', 'cosine_090']
        cls_func = [cosine_cls, cosine_cls, cosine_cls, cosine_cls]
        cls_thresholds = [0.25, 0.5, 0.75, 0.90]
        classifiers = list(zip(cls_names, cls_func, cls_thresholds))
        r_range = np.linspace(0, 5, 10) 
        print("R-range", r_range)
    elif args.param == "n":
        choice_methods = ['random']
        align_methods = ['s4a']
        cls_names = ['cosine_025', 'cosine_050', 'cosine_075', 'cosine_090']
        cls_func = [cosine_cls, cosine_cls, cosine_cls, cosine_cls]
        cls_thresholds = [0.25, 0.5, 0.75, 0.90]
        classifiers = list(zip(cls_names, cls_func, cls_thresholds))
        r_range [2.5]

        n_pos_range = np.arange(500, 5500, 500)
        print("N_pos range", n_pos_range)
        n_neg_range = np.arange(500, 5500, 500)
        print("N_neg range", n_neg_range)
    elif args.param == "choice_method":
        n_pos_range = [500]
        n_neg_range = [500]
        align_methods = ['s4a']
        cls_names = ['cosine_025', 'cosine_050', 'cosine_075', 'cosine_090']
        cls_func = [cosine_cls, cosine_cls, cosine_cls, cosine_cls]
        cls_thresholds = [0.25, 0.5, 0.75, 0.90]
        classifiers = list(zip(cls_names, cls_func, cls_thresholds))
        r_range = [2.5]
        choice_methods = ['random', 'far', 'close']

    if not args.no_semeval:
        for lang in languages:
            wv1, wv2, targets, y_true = read_semeval_data(lang, normalized)  
            targets_1 = targets_2 = targets

            # For SemEval we repeat `target` in targets_1 and targets_2
            results = run_experiments(wv1, wv2, targets_1, targets_2, y_true,
                            r_range, n_pos=n_pos_range, n_neg=n_neg_range, 
                            choice_method=choice_methods, align_method=align_methods,
                            classifier=classifiers)



            for h, r in results:
                print(lang, r)
                print('semeval_'+lang, normalized, *r, sep=',')
                print('semeval_'+lang, normalized, *r, sep=',', file=fout)
    
    if not args.no_ukus:
            wv1, wv2, targets, y_true = read_ukus_data(normalized)
            targets_1, targets_2 = zip(*targets)

            results = run_experiments(wv1, wv2, targets_1, targets_2, y_true,
                            r_range, n_pos=n_pos_range, n_neg=n_neg_range, 
                            choice_method=choice_methods, align_method=align_methods,
                            classifier=classifiers)
            for h, r in results:
                print("ukus", r)
                print("ukus", normalized, *r, sep=',')
                print("ukus", normalized, *r, sep=',', file=fout)           

    if not args.no_spanish:
        wv1, wv2, targets, y_true = read_spanish_data(normalized)
        targets_1 = targets_2 = targets

        results = run_experiments(wv1, wv2, targets_1, targets_2, y_true,
                        r_range, n_pos=n_pos_range, n_neg=n_neg_range, 
                        choice_method=choice_methods, align_method=align_methods,
                        classifier=classifiers)
        for h, r in results:
            print("spanish", r)
            print("spanish", normalized, *r, sep=",")
            print('spanish', normalized, *r, sep=',', file=fout)
    fout.close()


if __name__ == "__main__":
    new_main()
