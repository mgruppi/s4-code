import json
import os


if __name__ == '__main__':
    # Count the words in arxiv files
    path = '/data/corpus/arxiv-category-corpora/'
    files = os.listdir(path)

    token_count = 0
    types = set()
    for f in files[:1]:
        with open(os.path.join(path, f)) as fin:
            data = json.load(fin)
        for article in data:
            token_count += len(article)
            for token in article:
                types.add(token)

    print("Tokens:", token_count)
    print("Types:", len(types))
