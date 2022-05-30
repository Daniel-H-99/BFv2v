#!/bin/bash

. /home/server25/anaconda3/etc/profile.d/conda.sh

data_dir="/home/server25/minyeong_workspace/Experiment/data/demo/src"
output_dir="/home/server25/minyeong_workspace/Experiment/data/demo/src_SR"

mkdir -p $output_dir
for vid in $data_dir/*.jpg
do
    vid_name=$(basename "$vid" | sed 's/\(.*\)\..*/\1/')
    echo "working on $vid_name..."
    python crop-video.py --inp $vid --output $output_dir/${vid_name}.png
done