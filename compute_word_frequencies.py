from collections import defaultdict
import argparse
import spacy
import re
import os


def get_word_counts(sentences):
    """
    Returns the word counts given a list of tokenized sentences.

    Args:
        sentences(list[list[str]]) : List of tokenized sentences.
    
    Returns:
        counts(dict[str->int]) : Dictionary of word counts.
    """

    counts = defaultdict(int)

    for sent in sentences:
        for token in sent:
            counts[token] += 1
    
    return counts


def tokenize(lines, tokenizer='spacy'):
    """
    Tokenize input lines

    Args:
        lines(list[str]) : Input lines/sentences.
        tokenizer(str) : Tokenizer to use ('spacy', 'split'). 'split' will break tokens on '\s' characters.

    Returns:
        sentences(list[list[str]]) : List of tokenized sentences
    """            

    sentences = list()
    if tokenizer == 'spacy':
        nlp = spacy.load('en_core_web_sm')
        docs = nlp.pipe(lines)
        for doc in docs:
            sentences.append([t.text for t in doc])
    elif tokenizer == 'split':
        for line in lines:
            tokens = re.split("\s", line)
            sentences.append(tokens)
    else:
        raise NotImplementedError("Tokenizer not implemented")
    
    return sentences


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Path to input file.")
    parser.add_argument("--output", type=str, default=None, help="Path to output file.")
    parser.add_argument("--tokenizer", type=str, choices=["spacy", "split"],
                        default="spacy",
                        help="Which tokenizer to use.")

    args = parser.parse_args()


    if args.output is None:
        out_path = "results/word_counts"
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        args.output = os.path.join(out_path, os.path.basename(args.input))

    with open(args.input) as fin:
        lines = list(map(lambda s: s.strip(), fin.readlines()))
    
    sentences = tokenize(lines, args.tokenizer)
    counts = get_word_counts(sentences)

    with open(args.output, "w") as fout:
        for w in counts:
            fout.write("%s %s\n" % (w, counts[w]))
