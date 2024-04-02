import argparse
from WordVectors import WordVectors, intersection
from alignment import align
from scipy.spatial.distaince import cosine
from s4 import s4


if __name__ == "__main__":
    parser = argparse.ArgmentParser()
    parser.add_argument("path_src", help="Path to source vector files")
    parser.add_argument("path_tgt", help="Path to target vector files")

    parser.add_argument("--normalize", action="store_true", help="Normalize vectors")

    args = parser.parse_args()

    wv1 = WordVectors(input_file=args.path_src, normalized=args.normalize)
    wv2 = WordVectors(input_file=args.path_tgt, normalized=args.normalize)

    wv1, wv2 = intersection(wv1, wv2)

    landmarks, non_landmarks, Q = s4(wv1, wv2,
                                        cls_model="nn",
                                        n_targets=100,
                                        n_negatives=100,
                                        rate=1,
                                        t=0.5,
                                        iters=100,
                                        verbose=1,
                                        plot=1)
    wv1, wv2, Q = align(wv1, wv2, anchor_words=landmarks)
    d_l = [cosine(wv1[w], wv2[w]) for w in landmarks]
    d_n = [cosine(wv1[w], wv2[w]) for w in non_landmarks]

    with open("shift_distances.csv", "w") as fout:
        for w in wv1:
            d = cosine(wv1[w], wv2[w])
            fout.write(f"{w},{d}\n")
