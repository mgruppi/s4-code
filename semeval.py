"""
Run tests on SemEval2020 Task 1 data on the subtasks of:
    1 - binary classification
    2 - ranking
"""
import numpy as np
from WordVectors import WordVectors, intersection
from alignment import align
from s4 import s4, threshold_crossvalidation
from noise_aware import noise_aware

from scipy.spatial.distance import cosine, euclidean
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import defaultdict

import argparse

def get_feature_cdf(x):
    """
    Estimate a CDF for feature distribution x
    One way this can be done is via sorting arguments according to values,
    getting a sorted array of positions (low to high)
    then normalize this by len(x)
    Arguments:
        x       - feature vector
    Returns:
        p       - CDF values (percentile) for input feature vector
                i.e.: p[i] is the probability that X <= x[i]
    """
    y = np.argsort(x)
    p = np.zeros(len(x))
    for i, v in enumerate(y):
        p[v] = i+1  # i+1 is the position of element v in the CDF
    p = p/len(x)  # normalize for cumulative probabilities
    return p


def vote(x, hard=False):
    """
    Cast vote to decide whether there is semantic shift of a word or not.
    Arguments:
            x       - N x d array of N words and d features with columns as CDFs
            hard    - use hard voting, all features cast a binary vote, decision is averaged
                      if False, then votes are average, then binary the decision is made
    Returns:
            r       - Binary array of N elements (decision)
    """

    r = np.zeros((len(x)), dtype=float)
    for i, p in enumerate(x):
        if hard:
            p_vote = np.mean([float(pi > 0.5) for pi in p])
            r[i] = p_vote
        else:
            avg = np.mean(p)
            r[i] = avg
    return r


def main():
    """
    Performs tests on SemEval2020-Task 1 data on Unsupervised Lexical Semantic Change Detection.
    This experiments is designed to evaluate the performance of different landmark selection approaches,
    showing how the classification performance is affected by the landmark choices.
    """
    np.random.seed(1)

    align_methods = ["s4", "noise-aware", "top-10", "bot-10", "global",
                     "top-5", "bot-5"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", nargs="+",
                        help="Languages to use",
                        default=["english", "german", "latin", "swedish"])
    parser.add_argument("--cls", choices=["cosine", "s4", "cosine-auto"], default="cosine",
                        help="Classifier to use")

    args = parser.parse_args()
    languages = args.languages
    classifier = args.cls

    align_params = \
    {
        "english" : {
            "n_targets": 100,
            "n_negatives": 50,
            "rate": 1,
            "iters": 100
        },
        "german" : {
            "n_targets": 100,
            "n_negatives": 200,
            "rate": 1,
            "iters": 100
        },
        "latin" : {
            "n_targets": 10,
            "n_negatives": 4,
            "rate": 0.5,
            "iters": 100
        },
        "swedish" : {
            "n_targets": 100,
            "n_negatives": 200,
            "rate": 1,
            "iters": 100
        }
    }

    cls_params = \
    {
        "english": {
            "n_targets": 100,
            "n_negatives": 50,
            "rate": 1,
            "iters": 500
        },
        "german":{
            "n_targets": 50,
            "n_negatives": 200
        },
        "latin":
        {
            "n_targets": 50,
            "n_negatives": 10
        },
        "swedish":
        {
            "n_targets": 120,
            "n_negatives": 120
        }
    }

    auto_params = \
    {
        "english":
            {
            "rate": 1.5,
            "n_fold": 1,
            "n_targets": 50,
            "n_negatives": 100
            },
        "german":
        {
            "rate":1,
            "n_fold": 1,
            "n_targets": 200,
            "n_negatives": 100
        },
        "latin":
        {
            "rate": 1,
            "n_targets": 100,
            "n_negatives": 15
        },
        "swedish":
        {
            "rate": 1,
            "n_targets": 100,
            "n_negatives": 200
        }
    }

    normalized = False
    accuracies = defaultdict(dict)
    true_positives = defaultdict(dict)
    false_negatives = defaultdict(dict)
    correct_ans = defaultdict(dict)
    cm = defaultdict(dict)
    for lang in languages:
        # print("---")
        # print(lang)
        t = 0.5
        thresholds = np.arange(0.1, 1, 0.1)
        path_task1 = "data/semeval/truth/%s.txt" % lang
        path_task2 = "data/semeval/truth/%s.txt" % lang

        with open(path_task1) as fin:
            data = map(lambda s: s.strip().split("\t"), fin.readlines())
            targets, true_class = zip(*data)
            y_true = np.array(true_class, dtype=int)
        with open(path_task2) as fin:
            data = map(lambda s: s.strip().split("\t"), fin.readlines())
            _, true_ranking = zip(*data)
            true_ranking = np.array(true_ranking, dtype=float)

        corpus1_path = "wordvectors/semeval/%s-corpus1.vec" % lang
        corpus2_path = "wordvectors/semeval/%s-corpus2.vec" % lang
        wv1 = WordVectors(input_file=corpus1_path, normalized=normalized)
        wv2 = WordVectors(input_file=corpus2_path, normalized=normalized)

        c_method = defaultdict(list)
        wv1, wv2 = intersection(wv1, wv2)
        # print("Size of common vocab.", len(wv1))
        prediction = dict()  # store per-word prediction
        for align_method in align_methods:
            accuracies[align_method][lang] = list()
            true_positives[align_method][lang] = list()
            false_negatives[align_method][lang] = list()
            cm[align_method][lang] = np.zeros((2,2))


            if align_method == "global":
                landmarks = wv1.words
            elif align_method == "noise-aware":
                Q, alpha, landmarks, non_landmarks = noise_aware(wv1.vectors,
                                                                 wv2.vectors)
                landmarks = [wv1.words[i] for i in landmarks]
            elif align_method == "s4":
                landmarks, non_landmarks, Q = s4(wv1, wv2,
                                                                        cls_model="nn",
                                                                        verbose=0,
                                                                        **align_params[lang],

                                                                        )
            elif align_method == "top-10":
                landmarks = wv1.words[int(len(wv1.words)*0.1):]
            elif align_method == "top-5":
                landmarks = wv1.words[int(len(wv1.words)*0.05):]
            elif align_method == "top-50":
                landmarks = wv1.words[int(len(wv1.words)*0.50):]
            elif align_method == "bot-10":
                landmarks = wv1.words[-int(len(wv1.words)*0.1):]
            elif align_method == "bot-5":
                landmarks = wv1.words[-int(len(wv1.words)*0.05):]
            elif align_method == "bot-50":
                landmarks = wv1.words[-int(len(wv1.words)*0.50):]

            wv1_, wv2_, Q = align(wv1, wv2, anchor_words=landmarks)

            # Cosine-based classifier
            if classifier == "cosine":
                x = np.array([cosine(wv1_[w], wv2_[w]) for w in wv1.words])
                x = get_feature_cdf(x)
                x = np.array([x[wv1.word_id[i.lower()]] for i in targets])
                p = x.reshape(-1, 1)
                r = vote(p)
                y_pred = r

                best_acc = 0
                for t in thresholds:
                    y_bin = (y_pred>t)
                    correct = (y_bin == y_true)

                    accuracy = accuracy_score(y_true, y_bin)
                    if accuracy > best_acc:
                        prediction[align_method] = correct
                        best_acc = accuracy
                    tn, fp, fn, tp = confusion_matrix(y_true, y_bin).ravel()
                    cm[align_method][lang] += confusion_matrix(y_true, y_bin, normalize="all")
                    accuracies[align_method][lang].append(round(accuracy, 2))
                    true_positives[align_method][lang].append(round(tp, 2))
                    false_negatives[align_method][lang].append(round(fn, 2))
            elif classifier == "cosine-auto":
                t_cos = threshold_crossvalidation(wv1_, wv2_, iters=1,
                                                            **auto_params[lang],
                                                            landmarks=landmarks)
                x = np.array([cosine(wv1_[w], wv2_[w]) for w in wv1.words])
                x = get_feature_cdf(x)
                x = np.array([x[wv1.word_id[i.lower()]] for i in targets])
                p = x.reshape(-1, 1)
                r = vote(p)
                y_pred = r
                y_bin = y_pred > t_cos
                correct = (y_bin == y_true)

                accuracy = accuracy_score(y_true, y_bin)

                accuracies[align_method][lang].append(round(accuracy, 2))

            elif classifier == "s4":
                model = s4(wv1_, wv2_, landmarks=landmarks,
                                                    verbose=0,
                                                    **cls_params[lang],
                                                    update_landmarks=False)
                # Concatenate vectors of target words for prediction
                x = np.array([np.concatenate((wv1_[t.lower()], wv2_[t.lower()])) for t in targets])
                y_pred = model.predict(x)
                y_bin = y_pred > 0.5
                correct = (y_bin == y_true)

                accuracy = accuracy_score(y_true, y_bin)
                print(accuracy)
                accuracies[align_method][lang].append(round(accuracy, 2))


            c_method[align_method] = y_pred
            rho, pvalue = spearmanr(true_ranking, y_pred)



            # print(lang, align_method, "acc", accuracies[align_method][lang],
            #                                 "\nranking", round(rho, 2),
            #                                 "landmarks", len(landmarks))


    print("|Method|Language|Mean acc.|Max acc.|")
    print("|------|--------|---------|--------|")
    for method in accuracies:
        print("|",method, end="|")
        for lang in accuracies[method]:
            print(lang, round(np.mean(accuracies[method][lang]), 2), np.max(accuracies[method][lang]), sep="|", end="|\n")
    print()

if __name__ == "__main__":
    main()
