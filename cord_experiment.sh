#/bin/bash

# cat1="wordvectors/arxiv-bigram/arXiv_category_cs.AI.vec"
# cat2="wordvectors/arxiv-bigram/arXiv_category_physics.class-ph.vec"

cat1="../data/cord-19.vec"
cat2="../data/news-group3.vec"

# cat1="wordvectors/ukus/eng_uk_bnc.vec"
# cat2="wordvectors/ukus/eng_us_coca.vec"

# Compute semantic shifts
# python3 cord19.py $cat1 $cat2

# Produce table of results
# python3 arxiv_table.py "results/arxiv/cord-19.vec-news-group3.vec.csv" > "results/arxiv/table.md"
python3 arxiv_table.py "results/arxiv/cord-19.vec-coca-newspaper-all.vec.csv" > "results/arxiv/table.md"
