#!/bin/bash

TASK_NAME=$1
ID=$2
BASE_PATH=$3

#pip install requests
pip install medpy
#pip install cupy-cuda90

if [ $TASK_NAME == "segmentation" ];then
    python3 ./script/segmentation.py -in=$ID -bp=$BASE_PATH
elif [ $TASK_NAME == "merge" ];then
    python3 ./script/merge.py -in=$ID -bp=$BASE_PATH
# elif [ $TASK_NAME == "move_and_zeros" ];then
    # python3 ./scripts/move_and_zeros.py -in=$ID -bp=$BASE_PATH
# elif [ $TASK_NAME == "add_ids_2" ];then
    # python3 ./scripts/add_ids_2.py -in=$ID -bp=$BASE_PATH
# elif [ $TASK_NAME == "stitching" ];then
    # python2 ./scripts/pairwise_match.py -in=$ID -bp=$BASE_PATH
# elif [ $TASK_NAME == "concat" ];then
    # python3 ./scripts/concatenate_joins.py -in=$ID -bp=$BASE_PATH
# elif [ $TASK_NAME == "global" ];then
    # python3 ./scripts/create_global_map.py -in=$ID -bp=$BASE_PATH
# elif [ $TASK_NAME == "remap" ];then
    # python3 ./scripts/remap_block.py -in=$ID -bp=$BASE_PATH
else
    echo "There are no matching conditions."
fi
