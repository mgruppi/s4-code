"""
Runs parameter search for `r` in S4
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from WordVectors import WordVectors, intersection
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from s4 import s4, threshold_crossvalidation
from scipy.spatial.distance import cosine
from param_search_semeval import get_feature_cdf, vote
from alignment import align


def cosine_cls(wv1, wv2, targets, y_true, threshold=0.5, **kwargs):
    x = np.array([cosine(wv1[w], wv2[w]) for w in wv1.words])
    x = get_feature_cdf(x)
    x = np.array([x[wv1.word_id[i.lower()]] for i in targets])
    p = x.reshape(-1, 1)
    y = vote(p)
    y_pred = y

    y_bin = (y_pred > threshold)
    accuracy = accuracy_score(y_true, y_bin)
    precision = precision_score(y_true, y_bin)
    recall = recall_score(y_true, y_bin)
    f1 = f1_score(y_true, y_bin)

    return accuracy, precision, recall, f1


def s4_cls(wv1, wv2, targets, y_true, **kwargs):
    model = s4(wv1, wv2, update_landmarks=False,
               verbose=0,
               **kwargs)
    x = np.array([np.concatenate((wv1[t.lower()], wv2[t.lower()])) for t in targets])
    y_pred = model.predict(x)
    y_bin = y_pred > 0.5

    accuracy = accuracy_score(y_true, y_bin)
    precision = precision_score(y_true, y_bin)
    recall = recall_score(y_true, y_bin)
    f1 = f1_score(y_true, y_bin)

    return accuracy, precision, recall, f1


def run_experiments_semeval(wv1, wv2, targets, y_true, num_trials=10, r_upper=2,
                            cls=cosine_cls,
                            **kwargs):
    """
    Performs experiments by varying R in a range for a given input
    Args:
        num_trials:
        r_upper:

    Returns:

    """
    np.random.seed(1)
    r_range = np.linspace(0, r_upper, 2)
    print("R range", r_range)

    results = list()

    for r_ in r_range:
        for i in range(num_trials):

            landmarks, non_landmarks, Q, = s4(wv1, wv2,
                                              verbose=0,
                                              rate=r_,
                                              )
            wv1_, wv2_, Q = align(wv1, wv2, anchor_words=landmarks)
            acc, prec, rec, f1 = cls(wv1_, wv2_, targets, y_true, landmarks=landmarks, **kwargs)
            res_tuple = (r_, acc, prec, rec, f1)
            print(*res_tuple, sep=",")
            results.append(res_tuple)
    return results


def read_semeval_data(lang):
    # Load SemEval 2020 data
    corpus1_path = "wordvectors/semeval/%s-corpus1.vec" % lang
    corpus2_path = "wordvectors/semeval/%s-corpus2.vec" % lang
    normalized = False
    wv1 = WordVectors(input_file=corpus1_path, normalized=normalized)
    wv2 = WordVectors(input_file=corpus2_path, normalized=normalized)
    wv1, wv2 = intersection(wv1, wv2)

    path_task1 = "data/semeval/truth/%s.txt" % lang
    with open(path_task1) as fin:
        data = map(lambda s: s.strip().split("\t"), fin.readlines())
        targets, true_class = zip(*data)
        y_true = np.array(true_class, dtype=int)

    return wv1, wv2, targets, y_true


if __name__ == "__main__":

    languages = ["english", "german", "latin", "swedish"]

    fout_cosine = open("param_search_results_semeval_cosine.txt", "w")
    fout_s4 = open("param_search_results_semeval_s4.txt", "w")

    fout_cosine.write("language,r,accuracy,precision,recall,f1\n")
    fout_s4.write("language,r,accuracy,precision,recall,f1\n")
    for lang in languages:
        wv1, wv2, targets, y_true = read_semeval_data(lang)

        results_semeval = run_experiments_semeval(wv1, wv2, targets, y_true,
                                                  threshold=0.1
                                                  )
        for res in results_semeval:
            print(lang, *res, sep=",", file=fout_cosine)

        results_semeval_s4 = run_experiments_semeval(wv1, wv2, targets, y_true, cls=s4_cls)
        for res in results_semeval_s4:
            print(lang, *res, sep=",", file=fout_s4)

    fout_cosine.close()
    fout_s4.close()

