import argparse
from WordVectors import WordVectors, intersection
from alignment import align
from scipy.spatial.distance import cosine
from s4 import s4
from gensim.models import Word2Vec
from pathlib import Path


def load_word2vec(path, normalize):
    model = Word2Vec.load(path)
    words = list(model.wv.index_to_key)
    vectors = model.wv.vectors

    wv = WordVectors(words=words, vectors=vectors, normalized=normalize)
    return wv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_src", help="Path to source vector files")
    parser.add_argument("path_tgt", help="Path to target vector files")
    parser.add_argument("--iters", default=300, help="Num iterations")
    parser.add_argument("--n_targets", default=100, help="Num of targets to sample")
    parser.add_argument("--n_negatives", default=100, help="Number of negative samples")
    parser.add_argument("--rate", default=1, help="The rate of change")

    parser.add_argument("--normalize", action="store_true", help="Normalize vectors")

    args = parser.parse_args()

    wv1 = load_word2vec(args.path_src, args.normalize)
    wv2 = load_word2vec(args.path_tgt, args.normalize)
    # wv1 = WordVectors(input_file=args.path_src, normalized=args.normalize)
    # wv2 = WordVectors(input_file=args.path_tgt, normalized=args.normalize)

    wv1, wv2 = intersection(wv1, wv2)

    landmarks, non_landmarks, Q = s4(wv1, wv2,
                                        cls_model="nn",
                                        n_targets=args.n_targets,
                                        n_negatives=args.n_negatives,
                                        rate=args.rate,
                                        t=0.5,
                                        iters=args.iters,
                                        verbose=1,
                                        plot=0)
    wv1, wv2, Q = align(wv1, wv2, anchor_words=landmarks)
    d_l = [cosine(wv1[w], wv2[w]) for w in landmarks]
    d_n = [cosine(wv1[w], wv2[w]) for w in non_landmarks]

    file_src = Path(args.path_src)
    file_tgt = Path(args.path_tgt)


    with open(f"shift_distances_{file_src.name}_{file_tgt.name}.csv", "w") as fout:
        for w in wv1.words:
            d = cosine(wv1[w], wv2[w])
            fout.write(f"{w},{d}\n")
