#!/bin/bash

. /home/server25/anaconda3/etc/profile.d/conda.sh

data_dir="/home/ubuntu/workspace/mead_samples"
output_dir="/home/ubuntu/workspace/Experiment/data/mead"

mkdir -p $output_dir
for vid in $data_dir/*.mp4
do
    vid_name=$(basename "$vid" | sed 's/\(.*\)\..*/\1/')
    echo "working on $vid_name..."
    python crop-video.py --inp $vid --output $output_dir/${vid_name}.mp4
done