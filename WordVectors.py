import numpy as np
from collections import OrderedDict
from sklearn import preprocessing


# This file contains the WordVectors class used to load and handle word embeddings
def intersection(*args):
    """
    This function returns the intersection between WordVectors objects
    I.e.: all words that occur in both objects simultaneously as well as their
          respective word vectors
    Returns: list(WordVectors) objects with intersecting words
    """
    if len(args) < 2:
        print("! Error: intersection requires at least 2 WordVector objects")
        return None
    # Get intersecting words
    # WARNING: using set intersection will affect the order of words
    # in the original word vectors, to keep results consistent
    # it is better to iterate over the list of words
    # the resulting order will follow the first WordVectors's order
    # Get intersecting words
    common_words = set.intersection(*[set(wv.words) for wv in args])
    # Get intersecting words following the order of first WordVector
    words = [w for w in args[0].words if w in common_words]

    # Retrieve vectors from a and b for intersecting words
    wv_out = list()  # list of output WordVectors
    for wv in args:
        wv_out.append(WordVectors(words=words, vectors=[wv[w]for w in words]))

    return wv_out


def union(*args, f="average"):
    """
    Performs union of two or more word vectors, returning a new WordVectors
    containing union of words and combination of vectors according to given
    function.
    Arguments:
        *args   - list of WordVectors objects
        f       - (str) function to use when combining word vectors (default to average)
    Returns:
        wv      - WordVectors as the union the input args
    """

    if f == 'average':
        f = lambda x: sum(x)/len(x)

    union_words = set.union(*[set(wv.words) for wv in args])

    words = list(union_words)
    vectors = np.zeros((len(words), args[0].dimension), dtype=float)
    for i, w in enumerate(words):
        # Get list of existing vectors for w
        vecs = np.array([wv[w] for wv in args if w in wv])
        vectors[i] = f(vecs)  # Combine vectors

    wv_out = WordVectors(words=words, vectors=vectors)

    return wv_out


# Implements a WordVector class that performs mapping of word tokens to vectors
# Stores words as
class WordVectors:
    """
    WordVectors class containing methods for handling the mapping of words
    to vectors.
    Attributes
    - word_id -- OrderedDict mapping word to id in list of vectors
    - words -- list of words mapping id (index) to word string
    - vectors -- n x dim matrix of word vectors, follows id order
    - counts -- not used at the moment, designed to store word count
    - dimension -- dimension of wordvectors
    - zipped -- a zipped list of (word, vec) used to construct the object
    - min_freq -- filter out words whose frequency is less than min_freq
    """
    def __init__(self, words=None, vectors=None, counts=None, zipped=None,
                 input_file=None, centered=True, normalized=False,
                 min_freq=0, word_frequency=None):

        if words is not None and vectors is not None:
            self.word_id = OrderedDict()
            self.words = list()
            for i, w in enumerate(words):
                self.word_id[w] = i
            self.words = list(words)
            self.vectors = np.array(vectors)
            self.counts = counts
            self.dimension = len(vectors[0])
        elif zipped:
            pass
        elif input_file:
            self.dimension = 0
            self.word_id = dict()
            self.words = list()
            self.counts = dict()
            self.vectors = None
            self.read_file(input_file)

        if centered:
            self.center()
        if normalized:
            self.normalize()

        if word_frequency:
            self.filter_frequency(min_freq, word_frequency)

    def center(self):
        self.vectors = self.vectors - self.vectors.mean(axis=0, keepdims=True)

    def normalize(self):
        self.vectors = preprocessing.normalize(self.vectors, norm="l2")

    def get_words(self):
        return self.word_id.keys()

    # Returns a numpy (m, dim) array for a given list of words
    # I.e.: select vectors whose word are in argument words
    def get_vectors_from_words(self, words):
        vectors = np.zeros((len(words), self.dimension))
        for i, w in enumerate(words):
            vectors[i] = self[w]
        return vectors

    # Return (word,vec) for given word
    # In future versions may only return self.vectors
    def loc(self, word, return_word=False):
        if return_word:
            return word, self.vectors[self.word_id[word]]
        else:
            return self.vectors[self.word_id[word]]

    def get_count(self, word):
        return self.freq[self.word_id[word]]

    # Get word, vector pair from id
    def iloc(self, id_query, return_word=False):
        if return_word:
            return self.words[id_query], self.vectors[id_query]
        else:
            return self.vectors[id_query]

    # Overload [], given word w returns its vector
    def __getitem__(self, key):
        if isinstance(key, int) or isinstance(key, np.int64):
            return self.iloc(key)
        elif isinstance(key, slice):  # slice
            return ([w for w in self.words[key.start: key.stop]],
                    [v for v in self.vectors[key.start: key.stop]])
        return self.loc(key)

    def __len__(self):
        return len(self.words)

    def __contains__(self, word):
        return word in self.word_id

    def filter_frequency(self, min_freq, word_frequency):
        print("Filtering %d" % min_freq)
        words_kept = list()
        vectors_kept = list()
        for word, vec in zip(self.words, self.vectors):
            if word in word_frequency and word_frequency[word] > min_freq:
                words_kept.append(word)
                vectors_kept.append(vec)

        self.words = words_kept
        self.vectors = np.array(vectors_kept)
        self.word_id = OrderedDict()
        for i, w in enumerate(self.words):
            self.word_id[w] = i

        print(" - Found %d words" % len(self.words))

    # Read file in following format:
    # n_items dim
    def read_file(self, path):
        with open(path) as fin:
            n_words, dim = map(int, fin.readline().rstrip().split(" ", 1))
            self.dimension = dim
            # print("Reading WordVectors (%d,%d)" % (n_words, dim))

            # Use this function to process line reading in map
            def process_line(s):
                s = s.rstrip().split(" ", 1)
                w = s[0]
                v = np.array(s[1].split(" "), dtype=float)
                return w, v

            data = map(process_line, fin.readlines())
            self.words, self.vectors = zip(*data)
            self.words = list(self.words)
            self.word_id = {w: i for i, w in enumerate(self.words)}
            self.vectors = np.array(self.vectors, dtype=float)

    def save_txt(self, path):
        with open(path, "w") as fout:
            fout.write("%d %d\n" % (len(self.word_id), self.dimension))
            for word, vec in zip(self.words, self.vectors):
                v_string = " ".join(map(str, vec))
                fout.write("%s %s\n" % (word, v_string))
