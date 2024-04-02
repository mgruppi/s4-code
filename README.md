# Fake it Till You Make it: Self-Supervised Semantic Shifts for Monolingual Word Embedding Tasks

This code repository contains the code for the experiments seen in the paper  `Fake it Till You Make it: Self-Supervised Semantic Shifts for Monolingual Word Embedding Tasks` (2020).

## Requirements

Python version: Python 3.8. __Some of the dependencies are not compatible with newer Python versions__.

This repository contains mainly Python3 routines and dependencies listed in `requirements.txt`. To install the dependencies using pip/venv, run:

```
pip3 install -r requirements.txt
```

## Setup

After installing the requirements, run `setup.sh` to configure the environment and download the pre-trained word embeddings.
```
sh setup.sh
```
This will create the folders to store the results, and will download pre-trained vectors. The size of the download is approximately 500MB.

Alternatively, the pre-trained embeddings can be downloaded [here](https://zenodo.org/record/3890109/files/wordvectors.zip?download=1).


## Results


### British English :gb: vs. American English :us:

Results for the classification task on detecting semantic shift between British English and American English.

** Requires the pre-trained word embeddings from BNC and COCA **

To reproduce these results, run:

```
chmod +x ukus_experiment.sh
./ukus_experiment.sh
```

By default, results are saved to `results/ukus/cls_results.txt`.

|Method|Alignment|Accuracy|Precision|Recall|F1|
|------|---------|--------|---------|------|--|
|COS|global|0.35|0.71|0.19|0.3|
|S4-D|global|0.45 +- 0.02|0.45 +- 0.02|0.45 +- 0.02|0.45 +- 0.03||
|Noisy-Pairs|-|0.29|1.0|0.03|0.06|



### SemEval-2020 Task on Unsupervised Lexical Semantic Change Detection

Results for the binary classification task on semantic shift for multiple languages (SemEval2020 Task 1): English, German, Latin, and Swedish.

** Requires the pre-trained embeddings from SemEval **

To reproduce these results:

```
chmod +x semeval_experiment.sh
./semeval_experiment.sh
```

By default, results are saved to `results/semeval/cls_results.txt`.

|Method|Language|Mean acc.|Max acc.|
|------|--------|---------|--------|
| s4|english|0.62|0.7|
| noise-aware|english|0.61|0.65|
| top-10|english|0.59|0.68|
| bot-10|english|0.58|0.68|
| global|english|0.61|0.68|
| top-5|english|0.59|0.65|
| bot-5|english|0.57|0.68|



### ArXiv Semantic Shift Discovery

Word discovery experiment on the arXiv data set for subjects Artificial Intelligence (cs.AI) and Classical Physics (physics.class-ph). This table shows the list of top semantically shifted words uniquely discovered by Global, Noise-Aware and S4-A alignments, respectively. As well as the most shifted words commonly discovered by all three methods.

** Requires the pre-trained embeddings from arXiv **

To reproduce these results:

```
chmod +x arxiv_experiment.sh
./arxiv_experiment.sh
```

The table of results is saved in `results/arxiv/table.txt`, the ranking correlation plot is saved in `results/arxiv/arxiv_ranking.pdf`.

|Global|Noise-Aware|S4-A|Common| |
|------|-----------|----|------|-|
|agent||components|concepts|nodes|
|approximation||element|density|phys|
|boundary||mass|deterministic|polynomial|
|conceptual||order|die|probability|
|knowledge||solution|edge|respect|
|plane||space|equations|rev|
|reference||state|fields|rough|
|rules||term|internal|rule|
|system||time|light|tensor|
|systems||vector|los|variables|
