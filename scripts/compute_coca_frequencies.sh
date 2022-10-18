#!/bin/bash

input_path="/data/corpus/coca/coca/text"

for file in "$input_path"/*
do
    echo "$(basename $file)"
    python compute_word_frequencies.py $file --tokenizer split \
        --output "results/word_counts/coca/$(basename $file)"
done