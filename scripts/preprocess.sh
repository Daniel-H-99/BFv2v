#!/bin/bash

. /home/server25/anaconda3/etc/profile.d/conda.sh

data_dir="../YOUNG_v2"
output_dir="asset/young_v2"

for vid in $data_dir/*
do
    vid_name=$(basename "$vid" | sed 's/\(.*\)\..*/\1/')
    echo "working on $vid_name..."
    python crop-video.py --inp $vid --output $output_dir/${vid_name}_cropped.mp4
done