#/bin/bash

cat1="wordvectors/arxiv/arXiv_category_cs.AI.vec"
cat2="wordvectors/arxiv/arXiv_category_physics.class-ph.vec"

# cat1="wordvectors/ukus/eng_uk_bnc.vec"
# cat2="wordvectors/ukus/eng_us_coca.vec"

# Compute semantic shifts
python3 arxiv.py $cat1 $cat2

# Produce table of results
python3 arxiv_table.py "results/arxiv/cs.AI-physics.class-ph.csv" > "results/arxiv/table.md"
