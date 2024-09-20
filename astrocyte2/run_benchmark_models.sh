#!/bin/bash

declare -a arr=("Sparse" "Synchronous")

for x in "${arr[@]}"
do
  echo "Running $x model ..."
  python run_template.py "$x"
done

python plots.py
python makefig_benchmark_model.py
