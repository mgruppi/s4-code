#/bin/bash

# Make results directory
echo "- Creating directory structure"
mkdir "results"
mkdir "results/arxiv"
mkdir "results/ukus"
mkdir "results/semeval"

# Download pre-trained word vectors
echo "- Downloading pre-trained word vectors"
wget -O wordvectors.zip https://zenodo.org/record/3890109/files/wordvectors.zip?download=1

# Extract word embeddings
echo "- Extracting word embeddings"
unzip wordvectors.zip
rm wordvectors.zip


echo "- Setup complete - you can now run all the experiments with the pre-trained vectors"
