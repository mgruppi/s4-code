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


def perform_alignments(wv1, wv2, num_trials, choice_method,
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
        # cls_model = S4Network(wv1.dimension*2)  # This should not be the classifier from `_cls`, this is the internal S4 model
        print("Trial %d" % i)
        landmarks, non_landmarks, Q, =  s4(wv1, wv2,
                                            verbose=0,
                                            rate=r,
                                            n_targets=n_pos,
                                            n_negatives=n_neg,
                                            cls_model='nn',
                                            iters=iters, 
                                            inject_choice=choice_method
                                        )
        wv1_, wv2_, Q = align(wv1, wv2, anchor_words=landmarks)

        for w in wv1_.words:
            yield (i, w, cosine(wv1_[w], wv2_[w]))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('input_a', type=str, help="Input embeddings A")
    # parser.add_argument('input_b', type=str, help="Input embeddings B")
    # parser.add_argument("--normalized", action="store_true", help="Normalize word vectors")
    parser.add_argument("--num-trials", '--num_trials', dest="num_trials", type=int, default=3,
                        help="Number of trials.")
    parser.add_argument("--output_file", "--output-file", default=None, type=str, help="Path to output file.")

    args = parser.parse_args()
    args.normalized = True

    # Hard-coded pairs

    pairs = [
        ('wordvectors/semeval/english-corpus1.vec', 'wordvectors/semeval/english-corpus2.vec'),
        ('wordvectors/semeval/german-corpus1.vec', 'wordvectors/semeval/german-corpus2.vec'),
        ('wordvectors/semeval/latin-corpus1.vec', 'wordvectors/semeval/latin-corpus2.vec'),
        ('wordvectors/semeval/swedish-corpus1.vec', 'wordvectors/semeval/swedish-corpus2.vec'),
        ('wordvectors/spanish/spanish-old.vec', 'wordvectors/spanish/spanish-modern.vec'),
        ('wordvectors/ukus/bnc.vec', 'wordvectors/ukus/coca.vec'),
        ('wordvectors/arxiv/cs.AI.vec', 'wordvectors/arxiv/physics.class-ph.vec'),
        ('wordvectors/arxiv/cs.AI.vec', 'wordvectors/arxiv/cond-mat.mtrl-sci.vec'),
        ('wordvectors/arxiv/cs.AI.vec', 'wordvectors/arxiv/math.GM.vec'),
        ('wordvectors/arxiv/cs.AI.vec', 'wordvectors/arxiv/physics.atom-ph.vec'),
        ('wordvectors/coca/w_acad_2012.vec', 'wordvectors/coca/w_fic_2012.vec'),
        ('wordvectors/coca/w_acad_1990.vec', 'wordvectors/coca/w_fic_1990.vec'),
        ('wordvectors/coca/w_acad_2012.vec', 'wordvectors/coca/w_news_2012.vec'),
        ('wordvectors/coca/w_acad_2012.vec', 'wordvectors/coca/w_mag_2012.vec'),
        ('wordvectors/coca/w_acad_2012.vec', 'wordvectors/coca/w_spok_2012.vec'),
        ('wordvectors/coca/w_acad_2000.vec', 'wordvectors/coca/w_spok_2000.vec'),
        ('wordvectors/coca/w_acad_1990.vec', 'wordvectors/coca/w_spok_1990.vec')
    ]

    choice_methods = ['random', 'far', 'close']

    for a, b in pairs:
        args.input_a = a
        args.input_b = b
        try:

            if args.output_file is None:
                if not os.path.exists('results/choice_method'):
                    os.makedirs('results/choice_method')
                args.output_file = "results/choice_method/choice_method%s_%s" \
                                    % (os.path.basename(args.input_a).replace(".vec", ""), os.path.basename(args.input_b).replace(".vec", ""))
                if args.normalized:
                    args.output_file += "_normalized"
                args.output_file += ".csv"

            wv1 = WordVectors(input_file=args.input_a, normalized=args.normalized)
            wv2 = WordVectors(input_file=args.input_b, normalized=args.normalized)

            if not os.path.exists(os.path.dirname(args.output_file)):
                os.makedirs(os.path.dirname(args.output_file))

            with open(args.output_file, "w") as fout:
                fout.write("choice_method,trial_num,word,distance\n")
                for c in choice_methods:
                    results = perform_alignments(wv1, wv2, args.num_trials, c)
                    for res in results:
                        print(c, *res, sep=",", file=fout)
            
            
        except Exception as e:
            print("Error", e, a, b)
        args.output_file = None  # Clear output file before next pair
