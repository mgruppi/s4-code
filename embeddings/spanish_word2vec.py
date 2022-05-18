from gensim.models import Word2Vec
import os
from WordVectors import WordVectors

path_data = "/data/corpus/spanish/lchange22/corpora/"

path_old = os.path.join(path_data, "old_corpus/dataset_XIX_lemmatized.txt")
path_modern = os.path.join(path_data, "modern_corpus/modern_corpus_lemmatized.txt")

w2v_params = {
    "vector_size": 300,
    "window": 10,
    "min_count": 10,
    "workers": 64
}

if not os.path.exists("wordvectors/spanish"):
    os.makedirs("wordvectors/spanish")

model_old = Word2Vec(corpus_file=path_old, **w2v_params)
wv_old = WordVectors(words=model_old.wv.index_to_key, vectors=model_old.wv.vectors)

wv_old.save_txt("wordvectors/spanish/old.vec")

model_modern = Word2Vec(corpus_file=path_modern, **w2v_params)
wv_modern = WordVectors(words=model_modern.wv.index_to_key, vectors=model_modern.wv.vectors)

wv_modern.save_txt("wordvectors/spanish/modern.vec")