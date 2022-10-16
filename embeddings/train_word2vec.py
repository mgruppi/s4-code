import os
import spacy
import argparse
from multiprocessing import Pool
from gensim.models import Word2Vec
from WordVectors import WordVectors


def get_output_file_name(path):
    """
    Returns the output file name to avoid conflicting paths.
    If `path` already exists, returns the smallest path available by incrementing a suffix '_n'.

    Args:
        path(str) : The file path.
    
    Returns:
        p(str) : The valid path name. `None` if a valid name was not found.
    """

    if not os.path.exists(path):
        return path
    else:
        fname, ext = path.rsplit('.', 1)
        for count in range(1, 9999):  # Up to 9999 attempts
            fname_new = "%s_%d.%s" % (fname, count, ext)
            if not os.path.exists(fname_new):
                return fname_new
    return None


def process_doc(doc, pos_tag, lemmatize):
    """
    Process a single Doc.

    Args:
        doc(spacy.token.doc.Doc) : The spacy doc to be processed.
        pos_tag(bool) : Whether to apply pos tagging to the token.
        lemmatize(bool) : Whether to lemmatize the output token.
    
    Returns:
        sents(list[str]) : The tokenized sentences.
    """
    sents = list()
    for sent in doc.sents:
        tokens = list()
        for t in sent:
            tok = t.lemma_ if lemmatize else t.text
            if pos_tag:
                tok += "_"+t.pos_
            tokens.append(tok)
        sents.append(tokens)
    return sents


def preprocess_file(path, nlp, pos_tag=False, lemmatize=False):
    """
    Preprocess input file.

    Args:
        path(str) : Path to input corpus file.
        nlp(spacy.Model) : The spaCy model to use.
        pos_tag(bool) : If True, append the pos tag to the token with '_'. E.g.: string 'cat' becomes 'cat_noun' or corresponding pos tag.
        lemmatize(bool) : If True, tokens are replaced with the lemma (root).
    
    Returns:
        sentences(list[list[str]]) : The sentences from the input file in tokenized form.
    """

    print("Preprocessing text file...")

    with open(path) as fin:
        lines = map(lambda s: s.strip(), fin.readlines())
    
    docs = nlp.pipe(lines)

    print("  + Tokenizing...")
    sentences = list()
    for doc in docs:
        sentences.extend(process_doc(doc, pos_tag, lemmatize))
    
    return sentences


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input text file")
    parser.add_argument("--output_path", type=str, default=".", help="Output directory.")
    parser.add_argument("--pos_tag", "--pos-tag", action="store_true", help="Include POS tag to tokens.")
    parser.add_argument("--lemmatize", "--lemma", action="store_true", help="Lemmatize tokens.")
    parser.add_argument("--spacy_model", type=str, default="en_core_web_sm", help="SpaCy model to use.")

    args = parser.parse_args()

    nlp = spacy.load(args.spacy_model)
    # nlp.add_pipe("sentencizer")

    output_vec_file = os.path.basename(args.input).split(".")[0]
    if args.pos_tag:
        output_vec_file += "_pos"
    if args.lemmatize:
        output_vec_file += "_lemma"
    output_vec_file += ".vec"
    output_vec_file = os.path.join(args.output_path, output_vec_file)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    sentences = preprocess_file(args.input, nlp, args.pos_tag, args.lemmatize)
    w2v_params = {
        "vector_size": 300,
        "window": 10,
        "min_count": 10,
        "workers": 64
    }

    model = Word2Vec(sentences=sentences, **w2v_params)
    wv = WordVectors(words=model.wv.index_to_key, vectors=model.wv.vectors)

    output_vec_file = get_output_file_name(output_vec_file)
    print("Saving to", output_vec_file)

    wv.save_txt(output_vec_file)

    # with open(output_corpus_file, "w") as fout:
    #     for sent in sentences:
    #         fout.write("%s\n" % " ".join(sent))   


