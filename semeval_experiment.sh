#/bin/bash
# Run SemEval tests for all languages
# Reproduces Markdown table from README.md
languages=(
            "english"
            "german"
            "latin"
            "swedish"
          )

output="results/semeval/cls_results.txt"
echo "" > $output
for lang in ${languages[@]}
  do
    echo "$lang"
    python3 semeval.py --languages $lang >> $output
  done
