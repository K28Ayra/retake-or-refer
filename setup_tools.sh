#!/bin/sh
# Builds the conda env inside the repo. Run from the repo root.
set -e

conda env create -p envs/retina -f environment.yml
mkdir -p data/raw outputs results logs .kaggle
echo "done. run: conda activate ./envs/retina"
