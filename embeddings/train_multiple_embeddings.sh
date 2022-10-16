#!/bin/bash

num_runs=5
input_corpus="/data/corpus/coca/coca/category_aggregated/w_news.txt"
output_dir="wordvectors/isomorphism"

for i in $(seq $num_runs);
do
    echo "Run no. $i"
    python -m embeddings.train_word2vec "$input_corpus" --output_path "$output_dir"
done