#!/bin/bash
set -x
set -e

export CUDA_VISIBLE_DEVICES=0

data_dir=$1
prediction_dir=$2

if [[ $data_dir == *"dailylifeapis"* ]]; then
    dependency_type="temporal"
else
    dependency_type="resource"
fi

for file in $data_dir/$prediction_dir/*.json
do
    llm=$(basename $file .json)
    #  replace prediction_dir's "predictions" with "metrics"
    metrics=$(echo $prediction_dir | sed 's/predictions/metrics/g')
    if [ -f $data_dir/$metrics/${llm}_splits_all_tools_all_metric_all.json ] && [ -s $data_dir/$metrics/${llm}_splits_all_tools_all_metric_all.json ];
    then
        continue
    fi
    echo $llm
    python evaluate.py --data_dir $data_dir --prediction_dir $prediction_dir --llm $llm --splits all --n_tools all --mode add --dependency_type $dependency_type -m all 
done