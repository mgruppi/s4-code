import numpy as np
from collections import defaultdict


def read_file(path):
    freq_a = defaultdict(int)
    with open(path) as fin:
        for line in fin:
            l = line.strip().split(' ')
            if len(l) < 2:
                continue
            word, c = l
            c = int(c)
            freq_a[word] = c    
    
    return freq_a

print(" ===== SPANISH")
path_a = 'results/word_counts/spanish-dataset_XIX_lemmatized.txt'
path_b = 'results/word_counts/spanish-modern_corpus_lemmatized.txt'

freq_a = read_file(path_a)
freq_b = read_file(path_b)

total_a = np.sum([v for v in freq_a.values()])
total_b = np.sum([v for v in freq_b.values()])

words = ['primate', 'malayo', 'archivar', 'pedagogía']

for word in words:
    print("---", word)
    fa = freq_a[word]/total_a
    fb = freq_b[word]/total_b
    print(fa, fb, '| fa/fb =', fa/fb, fb/fa)
    print()


print(" ====== ENGLISH")
path_a = 'results/word_counts/english-corpus1.txt'
path_b = 'results/word_counts/english-corpus2.txt'

freq_a = read_file(path_a)
freq_b = read_file(path_b)

total_a = np.sum([v for v in freq_a.values()])
total_b = np.sum([v for v in freq_b.values()])

words = ['significantly', 'erection', 'lunatic', 'stint']

for word in words:
    print("---", word)
    fa = freq_a[word]/total_a
    fb = freq_b[word]/total_b
    print(fa, fb, '| fa/fb =', fa/fb, fb/fa)
    print()


print(" ====== COCA Academic")
path_a = 'results/word_counts/coca/w_acad_1990.txt'
path_b = 'results/word_counts/coca/w_acad_2012.txt'

freq_a = read_file(path_a)
freq_b = read_file(path_b)

total_a = np.sum([v for v in freq_a.values()])
total_b = np.sum([v for v in freq_b.values()])

words = ['emulated', 'reconstruction', 'nonhuman']

for word in words:
    print("---", word)
    fa = freq_a[word]/total_a
    fb = freq_b[word]/total_b
    print(fa, fb, '| fa/fb =', fa/fb, fb/fa)
    print()


print(" ====== ACADEMIC-NEWS(2012)")
path_a = 'results/word_counts/coca/w_acad_2012.txt'
path_b = 'results/word_counts/coca/w_news_2012.txt'

freq_a = read_file(path_a)
freq_b = read_file(path_b)

total_a = np.sum([v for v in freq_a.values()])
total_b = np.sum([v for v in freq_b.values()])

words = ['registration', 'reconstruction', 'masses', 'organ']

for word in words:
    print("---", word)
    fa = freq_a[word]/total_a
    fb = freq_b[word]/total_b
    print(fa, fb, '| fa/fb =', fa/fb, fb/fa)
    print()