import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
from WordVectors import WordVectors, intersection
import numpy as np
from s4 import s4
from s4_torch import S4Network
from scipy.spatial.distance import cosine
from alignment import align
import argparse


def perform_alignments(wv1, wv2, num_trials,
                        n_pos=200, n_neg=200, r=2, iters=100):
    """
    Perform alignment trials and predictions.

    Args:
        wv1(WordVectors) : WordVectors 1.
        wv2(WordVectors) : WordVectors 2.
        num_trials(int) : Number of trials.
    
    Returns:

    """

    wv1, wv2 = intersection(wv1, wv2)

    for i in range(num_trials):
        cls_model = S4Network(wv1.dimension*2)  # This should not be the classifier from `_cls`, this is the internal S4 model
        print("Trial %d" % i)
        landmarks, non_landmarks, Q, =  s4(wv1, wv2,
                                            verbose=0,
                                            rate=r,
                                            n_targets=n_pos,
                                            n_negatives=n_neg,
                                            cls_model=cls_model,
                                            iters=iters, 
                                        )
        wv1_, wv2_, Q = align(wv1, wv2, anchor_words=landmarks)

        for w in wv1_.words:
            yield (i, w, cosine(wv1_[w], wv2_[w]))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_a', type=str, help="Input embeddings A")
    parser.add_argument('input_b', type=str, help="Input embeddings B")
    parser.add_argument("--normalized", action="store_true", help="Normalize word vectors")
    parser.add_argument("--num-trials", '--num_trials', dest="num_trials", type=int, default=5,
                        help="Number of trials.")
    parser.add_argument("--output_file", "--output-file", default=None, type=str, help="Path to output file.")

    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = "results/direction/direction_%s_%s" \
                            % (os.path.basename(args.input_a).replace(".vec", ""), os.path.basename(args.input_b).replace(".vec", ""))
        if args.normalized:
            args.output_file += "_normalized"
        args.output_file += ".csv"

    wv1 = WordVectors(input_file=args.input_a, normalized=args.normalized)
    wv2 = WordVectors(input_file=args.input_b, normalized=args.normalized)

    if not os.path.exists(os.path.dirname(args.output_file)):
        os.makedirs(os.path.dirname(args.output_file))

    with open(args.output_file, "w") as fout:
        fout.write("direction,trial_num,word,distance\n")
        results = perform_alignments(wv1, wv2, args.num_trials)

        for res in results:
            print("a_to_b", *res, sep=",", file=fout)
        
        results = perform_alignments(wv2, wv1, args.num_trials)
        for res in results:
            print("b_to_a", *res, sep=",", file=fout)
