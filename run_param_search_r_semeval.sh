#!/bin/bash

rates=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2 2.1 2.2 2.3 2.4 2.5 3 3.5 4 4.5 5)

for r in ${rates[@]}
do
  echo "$r"
  python3 param_search_semeval.py --r "$r" >> "results_param_search_semeval.txt"
done