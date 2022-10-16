import argparse
import os
from alignment import align
from WordVectors import WordVectors, intersection
from scipy.spatial.distance import cosine
from noise_aware import noise_aware
from s4 import s4
from s4_torch import S4Network


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("wv_a", type=str, help="Path to word vectors a")
    # parser.add_argument("wv_b", type=str, help="Path to word vectors b")
    # args = parser.parse_args()

    input_pairs = (
        ("wordvectors/semeval/english-corpus1.vec", "wordvectors/semeval/english-corpus2.vec"),
        ("wordvectors/semeval/german-corpus1.vec", "wordvectors/semeval/german-corpus2.vec"),
        ("wordvectors/semeval/latin-corpus1.vec", "wordvectors/semeval/latin-corpus2.vec"),  
        ("wordvectors/semeval/swedish-corpus1.vec", "wordvectors/semeval/swedish-corpus2.vec"),  
        ("wordvectors/spanish/old.vec", "wordvectors/spanish/modern.vec"),
        ("wordvectors/ukus/bnc.vec", "wordvectors/ukus/coca.vec"),
    )

    output_path = "results/word_shifts"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    methods = ['global', 'noise-aware', 's4']

    for wv_a, wv_b in input_pairs:
        print(wv_a)
        print(wv_b)
        wv1 = WordVectors(input_file=wv_a, normalized=True)
        wv2 = WordVectors(input_file=wv_b, normalized=True)

        output = dict()
        for m in methods:
            print("    ", m)
            wva, wvb = intersection(wv1, wv2)

            if m == 'noise-aware':
                Q, alpha, landmarks, non_landmarks = noise_aware(wva.vectors, wvb.vectors)
            elif m == 's4':
                # cls_model = S4Network(wva.dimension*2)
                landmarks, non_landmarks, Q = s4(wva, wvb, rate=2, n_targets=100, n_negatives=100, cls_model='nn')
            elif m == 'global':
                landmarks = None

            wva, wvb, _Q = align(wva, wvb, anchor_words=landmarks)

            file_1 = os.path.basename(wv_a).rsplit(".", 1)[0]
            file_2 = os.path.basename(wv_b).rsplit(".", 1)[0]

            output_file = "%s-%s.txt" % (file_1, file_2)

            for word in wva.words:
                if word not in output:
                    output[word] = list()
                cos_dist = cosine(wva[word], wvb[word])
                output[word].append(cos_dist)

        with open(os.path.join(output_path, output_file), "w") as fout:
            fout.write("word global noise-aware s4\n")
            for word in output:
                fout.write("%s %s %s %s\n" % (word, *output[word]))

