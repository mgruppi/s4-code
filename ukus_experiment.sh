#/bin/bash

# Runs the British English vs American English experiment
# Output results into Markdown table as seen in README.md
alignments=("global"
          "top-5"
          "top-10"
          "bot-5"
          "bot-10"
          "s4"
          )

# Initializes results table
output="results/ukus/cls_results.txt"
echo "" > $output
echo "|Method|Alignment|Accuracy|Precision|Recall|F1|" >> $output
echo "|------|---------|--------|---------|------|--|" >> $output
for al in ${alignments[@]};
  do
    echo "alignment $al"
    python3 ukus.py $al >> $output --rounds 2
  done

echo "Done. Results saved in $output"
