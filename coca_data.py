import os
from collections import defaultdict


if __name__ == '__main__':
    path = '/data/corpus/coca/text/'

    category_count = defaultdict(int)

    for f in os.listdir(path):
        cat = f.rsplit('_', 1)[0]

        with open(os.path.join(path, f)) as fin:
            count = 0
            for line in fin:
                tokens = line.strip().split(' ')
                count += len(tokens)
            
        category_count[cat] += count
    

    for cat in category_count:
        print(' -', cat, category_count[cat])
    
    print(" = Total:", sum(category_count.values()))