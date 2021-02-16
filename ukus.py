# Runs all US vs UK english comparison experiments
import numpy as np
import argparse
from WordVectors import WordVectors, intersection
from alignment import align
from scipy.spatial.distance import cosine, euclidean
from noise_aware import noise_aware
from s4 import s4

from sklearn.metrics import accuracy_score, precision_score, recall_score, \
                            roc_auc_score, f1_score


def predict(x, y, method="cosine", t=0.5):
    if method == "cosine":
        return cosine (x, y) < t


def sample_dissimilar(n):
    """
    Generate dissimilar words randomly
    """


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("alignment", choices=['top-5', 'top-10', 'noise-aware', 'bot-5', 'bot-10', 'global', 's4'],
                        default="top",
                        help="Method to use in the alignment of UK to US")
    parser.add_argument("--rounds", type=int, default=1,
                        help="No. of rounds to run the classifications")

    args = parser.parse_args()

    path_us = "wordvectors/ukus/coca.vec"
    path_uk = "wordvectors/ukus/bnc.vec"
    path_dict = "data/ukus/dict_similar.txt"
    path_dict_dis = "data/ukus/dict_dissimilar.txt"

    normalized = False

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

    m = args.alignment
    # Align wordvectors (using any alignment approach)
    if m == "noise-aware":
        Q, alpha, landmarks, noise = noise_aware(wv_uk.vectors, wv_us.vectors)
        landmarks = [wv_uk.words[i] for i in landmarks]
        a_, b_, Q = align(wv_uk, wv_us, anchor_words=landmarks)
    elif m == "global":
        landmarks = wv_us.words
        a_, b_, Q = align(wv_uk, wv_us, anchor_words=landmarks)
        landmarks = landmarks[:len(landmarks)//2]
    elif m == "s4":
        landmarks = wv_us.words
        a_, b_, Q = align(wv_uk, wv_us, anchor_words=landmarks)
        landmarks, non_landmarks, Q = s4(wv_uk, wv_us,
                                            cls_model="nn",
                                            verbose=0,
                                            iters=100,
                                            n_targets=100,
                                            n_negatives=10,
                                            rate=0.25,
                                        )

        a_, b_, Q = align(wv_uk, wv_us, anchor_words=landmarks)
    elif m == "top-10":
        landmarks = wv_us.words[:int(len(wv_us.words)*0.1)]
    elif m == "top-5":
        landmarks = wv_us.words[:int(len(wv_us.words)*0.05)]
    elif m == "bot-10":
        landmarks = wv_us.words[-int(len(wv_us.words)*0.1):]
    elif m == 'bot-5':
        landmarks = wv_us.words[-int(len(wv_us.words)*0.05):]


    a_, b_, Q = align(wv_uk, wv_us, anchor_words=landmarks)

    wv1_ = WordVectors(words=wv1.words, vectors=np.dot(wv1.vectors, Q))

    test_pairs = dico
    # print("Landmarks", len(landmarks))
    # Train classifier
    self_scores = list()
    cos_scores = list()
    na_scores = list()
    iters=100

    # Interval to vary cosine thresholds
    cos_thresholds = [0.3, 0.5, 0.7]

    # Run several rounds, if given
    for r in range(args.rounds):
        model = s4(a_, b_, iters=iters, landmarks=landmarks,
                                            verbose=0,
                                            n_targets=1000,
                                            n_negatives=1000,
                                            rate=0.25,
                                            cls_model="nn",
                                            update_landmarks=False)

        acc = 0
        acc_cos = 0
        total = 0
        y_pred = list()
        y_pred_cos = list()
        try:
            x = np.array([np.concatenate((wv1_[p[0]], wv2[p[1]])) for p in test_pairs])
            x_cos = np.array([cosine(wv1_[p[0]], wv2[p[1]]) for p in test_pairs])

            # Predict with noise-aware
            # Generate pairs (u, v) and apply noise-aware
            # 0 if pair is clean, 1 if pair is noisy

            v_a = np.array([wv1_[p[0]] for p in test_pairs])
            v_b = np.array([wv2[p[1]] for p in test_pairs])
            Q, alpha, clean, noisy = noise_aware(v_a, v_b)

            y_pred_na = np.zeros((len(test_pairs)))
            for i in noisy:
                y_pred_na[i] = 1


        except KeyError as e:  # skip word if not in model
            pass
        y_hat = model.predict(x)
        y_pred = (y_hat > 0.5)

        self_acc = accuracy_score(y_true, y_pred)
        self_prec = precision_score(y_true, y_pred)
        self_rec = recall_score(y_true, y_pred)
        self_f1 = f1_score(y_true, y_pred)
        self_scores.append([self_acc, self_prec, self_rec, self_f1])

        # Cosine metrics
        # Compute average over multiple runs
        cos_acc = cos_prec = cos_rec = cos_f1 = 0
        for t in cos_thresholds:
            y_pred_cos = (x_cos > t)
            cos_acc = round(accuracy_score(y_true, y_pred_cos), 2)
            cos_prec = round(precision_score(y_true, y_pred_cos), 2)
            cos_rec = round(recall_score(y_true, y_pred_cos), 2)
            cos_f1 = round(f1_score(y_true, y_pred_cos), 2)

            cos_scores.append([cos_acc, cos_prec, cos_rec, cos_f1])

        # Noise-Aware metrics
        na_acc = round(accuracy_score(y_true, y_pred_na), 2)
        na_prec = round(precision_score(y_true, y_pred_na), 2)
        na_rec = round(recall_score(y_true, y_pred_na), 2)
        na_f1 = round(f1_score(y_true, y_pred_na), 2)
        na_scores.append([na_acc, na_prec, na_rec, na_f1])

    self_scores = np.array(self_scores)
    cos_scores = np.array(cos_scores)
    na_scores = np.array(na_scores)


    # Print Markdown Table
    for j, t in enumerate(cos_thresholds):
        print("|COS %.2f" % t, m, sep="|", end="|")
        for i in range(4):
            print("%.2f" % (round(cos_scores[j:, i].mean(), 2)), end="|", sep=" ")
        print("|")
    print("|")
    print("|S4-D", m, end="|", sep="|")
    for i in range(4):
        print("%.2f +- %.2f" %(round(self_scores[:, i].mean(), 2), round(self_scores[:, i].std(), 2)), end="|", sep=" ")
    print("|")
    print("|Noisy-Pairs", "-", *na_scores[0], sep="|", end="|\n")


if __name__ == "__main__":
    main()
