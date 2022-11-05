from gensim.models import Word2Vec
import os
from WordVectors import WordVectors
import spacy
import re
from multiprocessing import Pool


def tokenize(lines):
    sentences = list()

    for line in lines:
        tokens = [t for t in re.split("\s", line) if t != "@"]  # Remove '@' filler characters
        sentences.append(tokens)
    
    return sentences


def word2vec_job(corpus_path, w2v_params, output_dir):
    """
    Trains a word2vec model from input `corpus_path`.
    
    Args:
        corpus_path(str) : Path to input file.
        wv2_params(dict) : Word2Vec parameters.
        output_dir(str) : Path to output dir.
    
    """
    print("* Starting", corpus_path)

    try:
        with open(corpus_path) as fin:
            lines = fin.readlines()
        sentences = tokenize(lines)

        model = Word2Vec(sentences=sentences, **w2v_params)
        wv = WordVectors(words=model.wv.index_to_key, vectors=model.wv.vectors)

        output_basename = os.path.basename(corpus_path).replace(".txt", ".vec")
        wv.save_txt(os.path.join(output_dir, output_basename))
        print("    + Done", corpus_path)
    except UnicodeDecodeError as e:
        print("Decode error", e, corpus_path)

    return True


if __name__ == "__main__":
    # input_path = "/data/corpus/coca/coca/text/"
    # output_dir = "wordvectors/coca/"

    # # COHA paths
    input_path = '/data/corpus/coha/coha-wlp/'
    output_dir = 'wordvectors/coha/'

    w2v_params = {
        "vector_size": 300,
        "window": 10,
        "min_count": 10,
        "workers": 64
    }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    files = os.listdir(input_path)

    job_params = [(os.path.join(input_path, f), w2v_params, output_dir) for f in files]

    ### Multiprocessing
    with Pool() as p:
        results = p.starmap(word2vec_job, job_params)

    print("- All done")

