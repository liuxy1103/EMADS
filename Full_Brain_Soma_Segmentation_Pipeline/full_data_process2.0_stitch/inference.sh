#!/bin/bash

python ./scripts/block_inference.py -ib=$1 -bp=$2 -nu=$3 -mp=$4
python ./scripts/inference_tf_x4.py -ib=$1 -bp=$2 -nu=$3
