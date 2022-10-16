from parameter_search import read_semeval_data, read_spanish_data, read_ukus_data
import json


def main():
    metadata = dict()
    languages = ['english', 'german', 'latin', 'swedish']
    for lang in languages:
        ds = "semeval_%s" % lang
        wv1, wv2, targets, y_true = read_semeval_data(lang)
        metadata[ds] = dict()
        metadata[ds]['common_vocab_size'] = len(wv1.words)
        metadata[ds]['num_targets'] = len(targets)
    
    # UKUS
    wv1, wv2, targets, y_true = read_ukus_data()
    ds = "ukus"
    metadata[ds] = dict()
    metadata[ds]['common_vocab_size'] = len(wv1.words)
    metadata[ds]['num_targets'] = len(targets)

    # Spanish
    wv1, wv2, targets, y_true = read_spanish_data()
    ds = "spanish"
    metadata[ds] = dict()
    metadata[ds]['common_vocab_size'] = len(wv1.words)
    metadata[ds]['num_targets'] = len(targets)

    with open('data/metadata.json', 'w') as fout:
        json.dump(metadata, fout)

if __name__ == "__main__":
    main()
