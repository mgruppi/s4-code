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
from scipy.spatial.distance import cosine, euclidean
from param_search_semeval import get_feature_cdf, vote
from alignment import align
import argparse
import itertools


def cosine_cls(wv1, wv2, targets_1, targets_2, y_true, threshold=0.5, **kwargs):
    """
    Applies the cosine classifier to the data.

    Args:
        wv1(WordVectors) : Input word vectors
        wv2(WordVectors) : Input word vectors
        targets_1(list[str]) : Target words in wv1
        targets_2(list[str]) : Target words in wv2
        y_true(np.array[int]) : True labels
        threshold(float) : Classification threshold
        **kwargs(dict) : Keyword arguments
    
    Returns:
        accuracy(float) : Accuracy score
        precision(float) : Precision score
        recall(float) : Recall score
        f1(float) : F1 score
        tn(int) : True negatives
        fp(int) : False positives
        fn(int) : False negatives
        tp(int) : True positives
        correct(list[str]) : List of correctly predicted words
        incorrect(list[str]) : List of incorrectly predicted words
    """
    x = np.ones(len(targets_1))
    for i in range(len(targets_1)):
        t1, t2 = targets_1[i], targets_2[i]
        if t1.lower() in wv1 and t2.lower() in wv2:
            x[i] = cosine(wv1[t1.lower()], wv2[t2.lower()])
    y_pred = x.reshape(-1, 1)  # Prediction score

    y_bin = (y_pred > threshold)  # Get the binary prediction
    accuracy = accuracy_score(y_true, y_bin)
    precision = precision_score(y_true, y_bin)
    recall = recall_score(y_true, y_bin)
    f1 = f1_score(y_true, y_bin)
    tn, fp, fn, tp = confusion_matrix(y_true, y_bin).ravel()

    # Get correctly and incorrectly predicted words

    correct_idx = np.where(y_bin.flatten() == y_true)[0]
    incorrect_idx = np.where(y_bin.flatten() != y_true)[0]
    correct = [targets_1[i] for i in correct_idx]
    incorrect = [targets_1[i] for i in incorrect_idx]

    return accuracy, precision, recall, f1, tn, fp, fn, tp, correct, incorrect


def euclidean_cls(wv1, wv2, targets_1, targets_2, y_true, threshold=0.5, **kwargs):
    """
    Applies the cosine classifier to the data.

    Args:
        wv1(WordVectors) : Input word vectors
        wv2(WordVectors) : Input word vectors
        targets_1(list[str]) : Target words in wv1
        targets_2(list[str]) : Target words in wv2
        y_true(np.array[int]) : True labels
        threshold(float) : Classification threshold
        **kwargs(dict) : Keyword arguments
    
    Returns:
        accuracy(float) : Accuracy score
        precision(float) : Precision score
        recall(float) : Recall score
        f1(float) : F1 score
        tn(int) : True negatives
        fp(int) : False positives
        fn(int) : False negatives
        tp(int) : True positives
        correct(list[str]) : List of correctly predicted words
        incorrect(list[str]) : List of incorrectly predicted words
    """
    x = np.ones(len(targets_1))
    for i in range(len(targets_1)):
        t1, t2 = targets_1[i], targets_2[i]
        if t1.lower() in wv1 and t2.lower() in wv2:
            x[i] = euclidean(wv1[t1.lower()], wv2[t2.lower()])
    y_pred = x.reshape(-1, 1)  # Prediction score

    y_bin = (y_pred > threshold)  # Get the binary prediction
    accuracy = accuracy_score(y_true, y_bin)
    precision = precision_score(y_true, y_bin)
    recall = recall_score(y_true, y_bin)
    f1 = f1_score(y_true, y_bin)
    tn, fp, fn, tp = confusion_matrix(y_true, y_bin).ravel()

    # Get correctly and incorrectly predicted words

    correct_idx = np.where(y_bin.flatten() == y_true)[0]
    incorrect_idx = np.where(y_bin.flatten() != y_true)[0]
    correct = [targets_1[i] for i in correct_idx]
    incorrect = [targets_1[i] for i in incorrect_idx]

    return accuracy, precision, recall, f1, tn, fp, fn, tp, correct, incorrect


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


def filter_pos_tags(wv, filter_pos, targets={}):
    """
    Filter POS tags from WordVectors object.
    Requires WV to have appended POS tags in words.
    Only those POS tags in `filter_pos`are kept.

    Args:
        wv(WordVectors.WordVectors) : WordVectors to be modified.
        filter_pos(set or iterable) : POS tags to filter.
        targets(iterable) : Set of target words to keep regardless of POS tag.
    
    Returns:
        wv_f(WordVectors.WordVectors) : Filtered WordVectors.
    """

    # If filter_pos is invalid, return original wv.
    if len(filter_pos) == 0 or filter_pos is None:
        return wv
    
    words = list()
    vectors = list()

    targets = set(targets)

    for w, v in zip(wv.words, wv.vectors):
        splits = w.rsplit("_", 1)  # Using rsplit to split the rightmost '_'.
        if len(splits) < 2:
            continue
        word, pos = splits

        if pos in filter_pos or word in targets:
            words.append(word)
            vectors.append(v)
  
    wv_f = WordVectors(words=words, vectors=vectors)

    return wv_f


def read_semeval_data(lang, normalized=False, pos_lemma=False, filter_pos=None):
    # Load SemEval 2020 data
    if not pos_lemma:
        corpus1_path = "wordvectors/semeval/%s-corpus1.vec" % lang
        corpus2_path = "wordvectors/semeval/%s-corpus2.vec" % lang
    else:
        corpus1_path = "wordvectors/semeval/%s-corpus1_pos_lemma.vec" % lang
        corpus2_path = "wordvectors/semeval/%s-corpus2_pos_lemma.vec" % lang

    wv1 = WordVectors(input_file=corpus1_path, normalized=normalized)
    wv2 = WordVectors(input_file=corpus2_path, normalized=normalized)

    path_task1 = "data/semeval/truth/%s.txt" % lang
    with open(path_task1) as fin:
        data = map(lambda s: s.strip().split("\t"), fin.readlines())
        targets, true_class = zip(*data)
        y_true = np.array(true_class, dtype=int)    

    if filter_pos and pos_lemma:
        wv1 = filter_pos_tags(wv1, filter_pos, targets=targets)
        wv2 = filter_pos_tags(wv2, filter_pos, targets=targets)

    wv1, wv2 = intersection(wv1, wv2)

    return wv1, wv2, targets, y_true


def read_ukus_data(normalized=False, pos_lemma=False, filter_pos=None):

    if not pos_lemma:
        path_us = "wordvectors/ukus/coca.vec"
        path_uk = "wordvectors/ukus/bnc.vec"
    else:
        path_us = "wordvectors/ukus/coca_pos_lemma.vec"
        path_uk = "wordvectors/ukus/bnc_pos_lemma.vec"

    path_dict = "data/ukus/dict_similar.txt"
    path_dict_dis = "data/ukus/dict_dissimilar.txt"

    wv1 = WordVectors(input_file=path_uk, normalized=normalized)
    wv2 = WordVectors(input_file=path_us, normalized=normalized)

    # Load dictionaries of words
    with open(path_dict) as fin:
        dico_sim = list(map(lambda s: s.strip().split(" ", 1), fin.readlines()))

    with open(path_dict_dis) as fin:
        dico_dis = list(map(lambda s: (s.strip(), s.strip()), fin.readlines()))

    # Concatenate targets into a single list
    ta, tb = zip(*(dico_sim + dico_dis))
    targets = set(ta + tb)

    if filter_pos and pos_lemma:
        wv1 = filter_pos_tags(wv1, filter_pos, targets=targets)
        wv2 = filter_pos_tags(wv2, filter_pos, targets=targets)

    wv_uk, wv_us = intersection(wv1, wv2)

    # Filter words not in the vocabulry of either UK or US corpora
    dico_sim = [(a, b) for a, b in dico_sim if a in wv_uk.word_id and b in wv_us.word_id]
    dico_dis = [(a, b) for a, b in dico_dis if a in wv_uk.word_id and b in wv_us.word_id]
    dico = dico_sim + dico_dis

    # Create true labels for terms
    # 0 -> similar | 1 -> dissimilar
    y_true = [0] * len(dico_sim) + [1]*len(dico_dis)

    return wv_uk, wv_us, dico, y_true


def read_spanish_data(normalized=False, truth_column="change_binary", pos_lemma=False, filter_pos=None):
    """
    Reads spanish word vectors + ground truth data from the LChange2022 Shared Task.
    Arguments:
        normalized: Boolean value for reading normalized WordVectors
        truth_column: Name of the truth column to use. Defaults to "change_binary".
                        Other options are "change_binary_gain" and "change_binary_loss".
    """

    if not pos_lemma:
        path_old = "wordvectors/spanish/old.vec"
        path_modern = "wordvectors/spanish/modern.vec"
    else:
        path_old = "wordvectors/spanish/old_pos_lemma.vec"
        path_modern = "wordvectors/spanish/modern_pos_lemma.vec"

    wv1 = WordVectors(input_file=path_old, normalized=normalized)
    wv2 = WordVectors(input_file=path_modern, normalized=normalized)

    df = pd.read_csv("data/spanish/stats_groupings.csv", sep="\t")
    targets = df["lemma"]
    y_true = df[truth_column]

    if filter_pos and pos_lemma:
        wv1 = filter_pos_tags(wv1, filter_pos, targets=targets)
        wv2 = filter_pos_tags(wv2, filter_pos, targets=targets)

    wv_old, wv_mod = intersection(wv1, wv2)

    return wv_old, wv_mod, targets, y_true


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
        print("Running", _r, _np, _nn, _cm, _am, _cls)

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
                                                    iters=100, 
                                                )
            n_landmarks = len(landmarks)
            header_tuple = ('num_trial', 'r', 'n_pos', 'n_neg', 'choice_method', 'alignment', 'cls', 'landmarks', 'accuracy', 'precision',
                'recall', 'f1', 'true_negatives', 'false_positives', 'false_negatives', 'true_positives')
            if n_landmarks > 0:
                wv1_, wv2_, Q = align(wv1, wv2, anchor_words=landmarks)
                acc, prec, rec, f1, tn, fp, fn, tp, correct, incorrect = _cls[1](wv1_, wv2_, targets_1, targets_2, y_true, landmarks=landmarks, threshold=_cls[2])
                res_tuple = (i, _r, _np, _nn, _cm, _am, _cls[0], n_landmarks, acc, prec, rec, f1, tn, fp, fn, tp)
            else:
                res_tuple = (i, _r, _np, _nn, _cm, _am, _cls[0], n_landmarks, -1, -1, -1, -1, -1, -1, -1, -1)
            yield (header_tuple, res_tuple, correct, incorrect)


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
    parser.add_argument("--num-trials", '--num_trials', dest="num_trials", type=int, default=10,
                        help="Number of trials per r value")
    parser.add_argument("--normalized", action="store_true", help="Normalize word vectors")
    parser.add_argument("--languages", default=None, nargs="+", help="List of languages")
    parser.add_argument("--r-upper", dest="r_upper", default=2, type=float, help="Upper bound for r")
    parser.add_argument("--r-steps", dest="r_steps", default=11, type=int, help="No. of steps for r")
    parser.add_argument("--euclidean", action="store_true", help="Use euclidean distance instead of cosine distance.")
    parser.add_argument("--flip-direction", dest="flip_direction", action="store_true", help="Run S4 in reverse direction")
    parser.add_argument("--output-file", dest='output_file', default=None, type=str,
                        help='Change default output file')
    parser.add_argument("--pos_lemma", action="store_true", help="Open the pos_lemma version of the embeddings.")
    parser.add_argument("--filter_pos", default={"NOUN", "VERB"}, nargs="+", help="List of POS tags to keep in word vectors. E.g., use --filter_pos NOUN VERB to keep only nouns and verbs.")
                    

    args = parser.parse_args()

    normalized = args.normalized
    pos_lemma = args.pos_lemma
    filter_pos = args.filter_pos

    if args.languages is None:
        languages = ["english", "german", "latin", "swedish"]
    else:
        languages = args.languages
    
    if args.output_file is None:
        if not os.path.exists("results/"):
            os.makedirs("results/")

        datasets = list()
        if not args.no_semeval:
            datasets.append("semeval")
        if not args.no_ukus:
            datasets.append("ukus")
        if not args.no_spanish:
            datasets.append("spanish")
        args.output_file = "results/results_param_search_%s" % "+".join(datasets)
        if args.euclidean:
            args.output_file += "_euclidean"
        if args.normalized:
            args.output_file += '_normalized'
        if args.pos_lemma:
            args.output_file += '_pos_lemma'
        if args.flip_direction:
            args.output_file += '_inverse'
        args.output_file += '.csv'

        predictions_file = args.output_file.replace("results_", "predictions_")


    # Make file header
    header_tuple = (
    'dataset', 'normalized', 'flipped', 'num_trial', 'r', 'n_pos', 'n_neg', 'choice_method', 'alignment', 'cls', 'landmarks',
    'accuracy', 'precision', 'recall', 'f1', 
    'true_negatives', 'false_positives', 'false_negatives', 'true_positives')

    fout = open(args.output_file, 'w')
    fpred = open(predictions_file, "w")

    print(*header_tuple, sep=',', file=fout)
    print("dataset,word,correct,r", file=fpred)

    if args.param == "all":  # all vs all comparison
        n_pos_range = np.arange(500, 5500, 500)
        print("N_pos range", n_pos_range)
        n_neg_range = np.arange(500, 5500, 500)
        print("N_neg range", n_neg_range)

        r_range = np.linspace(0, args.r_upper, args.r_steps)
        print("R-range", r_range)

        choice_methods = ['random', 'far', 'close']
        print("Choice methods", choice_methods)

        align_methods = ['s4a']

        cls_names = ['cosine_025', 'cosine_050', 'cosine_075', 'cosine_090']
        cls_func = [cosine_cls, cosine_cls, cosine_cls, cosine_cls]
        cls_thresholds = [0.25, 0.5, 0.75, 0.90]
        classifiers = list(zip(cls_names, cls_func, cls_thresholds))
    elif args.param == "r":
        n_pos_range = [200]
        n_neg_range = [200]
        choice_methods = ['random']
        align_methods = ['s4a']
        cls_names = ['cosine_025', 'cosine_050', 'cosine_075', 'cosine_090']
        if not args.euclidean:
            cls_names = ['cosine_025', 'cosine_050', 'cosine_075', 'cosine_090']
            cls_func = [cosine_cls, cosine_cls, cosine_cls, cosine_cls]
            cls_thresholds = [0.25, 0.5, 0.75, 0.90]
        else:
            cls_names = ['euclidean_050', 'euclidean_100', 'euclidean_150', 'euclidean_200']
            cls_func = [euclidean_cls, euclidean_cls, euclidean_cls, euclidean_cls]
            cls_thresholds = [0.5, 1.0, 1.5, 2.0]
        classifiers = list(zip(cls_names, cls_func, cls_thresholds))
        r_range = np.linspace(0, args.r_upper, args.r_steps) 
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
        n_pos_range = [200]
        n_neg_range = [200]
        align_methods = ['s4a']
        if not args.euclidean:
            cls_names = ['cosine_025', 'cosine_050', 'cosine_075', 'cosine_090']
            cls_func = [cosine_cls, cosine_cls, cosine_cls, cosine_cls]
            cls_thresholds = [0.25, 0.5, 0.75, 0.90]
        else:
            cls_names = ['euclidean_050', 'euclidean_100', 'euclidean_150', 'euclidean_200']
            cls_func = [euclidean_cls, euclidean_cls, euclidean_cls, euclidean_cls]
            cls_thresholds = [0.5, 1.0, 1.5, 2.0]
        classifiers = list(zip(cls_names, cls_func, cls_thresholds))
        r_range = [2.5]
        choice_methods = ['random', 'far', 'close']

    if not args.no_semeval:
        for lang in languages:
            if not args.flip_direction:
                wv1, wv2, targets, y_true = read_semeval_data(lang, normalized, pos_lemma=pos_lemma, filter_pos=filter_pos)  
                targets_1 = targets_2 = targets
            else:
                wv2, wv1, targets, y_true = read_semeval_data(lang, normalized, pos_lemma=pos_lemma, filter_pos=filter_pos)
                targets_1 = targets_2 = targets


            # For SemEval we repeat `target` in targets_1 and targets_2
            results = run_experiments(wv1, wv2, targets_1, targets_2, y_true,
                            r_range, n_pos=n_pos_range, n_neg=n_neg_range, 
                            choice_method=choice_methods, align_method=align_methods,
                            classifier=classifiers)



            for h, r, correct, incorrect in results:
                print(lang, r)
                print('semeval_'+lang, normalized, args.flip_direction, *r, sep=',')

                print('semeval_'+lang, normalized, args.flip_direction, *r, sep=',', file=fout)

                # Write correct and incorrect words
                for word in correct:
                    print("semeval_"+lang, word, 1, r[1], sep=',', file=fpred)
                for word in incorrect:
                    print("semeval_"+lang, word, 0, r[1], sep=',', file=fpred)
    
    if not args.no_ukus:
            if not args.flip_direction:
                wv1, wv2, targets, y_true = read_ukus_data(normalized, pos_lemma=pos_lemma, filter_pos=filter_pos)
                targets_1, targets_2 = zip(*targets)
            else:
                wv2, wv1, targets, y_true = read_ukus_data(normalized, pos_lemma=pos_lemma, filter_pos=filter_pos)
                targets_2, targets_1 = zip(*targets)

            results = run_experiments(wv1, wv2, targets_1, targets_2, y_true,
                            r_range, n_pos=n_pos_range, n_neg=n_neg_range, 
                            choice_method=choice_methods, align_method=align_methods,
                            classifier=classifiers)
            for h, r, correct, incorrect in results:
                print("ukus", r)
                print("ukus", normalized, args.flip_direction, *r, sep=',')
                print("ukus", normalized, args.flip_direction, *r, sep=',', file=fout)

                # Write correct and incorrect words
                for word in correct:
                    print("ukus", word, 1, r[1], sep=',', file=fpred)
                for word in incorrect:
                    print("ukus", word, 0, r[1], sep=',', file=fpred)           

    if not args.no_spanish:
        if not args.flip_direction:
            wv1, wv2, targets, y_true = read_spanish_data(normalized, pos_lemma=pos_lemma, filter_pos=filter_pos)
            targets_1 = targets_2 = targets
        else:
            wv2, wv1, targets, y_true = read_spanish_data(normalized, pos_lemma=pos_lemma, filter_pos=filter_pos)
            targets_1 = targets_2 = targets

        results = run_experiments(wv1, wv2, targets_1, targets_2, y_true,
                        r_range, n_pos=n_pos_range, n_neg=n_neg_range, 
                        choice_method=choice_methods, align_method=align_methods,
                        classifier=classifiers)
        for h, r, correct, incorrect in results:
            print("spanish", r)
            print("spanish", normalized, args.flip_direction, *r, sep=",")
            print('spanish', normalized, args.flip_direction, *r, sep=',', file=fout)

            # Write correct and incorrect words
            for word in correct:
                print("spanish", word, 1, r[1], sep=',', file=fpred)
            for word in incorrect:
                print("spanish", word, 0, r[1], sep=',', file=fpred)

    fout.close()
    fpred.close()

if __name__ == "__main__":
    new_main()
