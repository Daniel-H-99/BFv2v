#!/bin/bash

# for file in /home/server25/minyeong_workspace/YOUNG/cropped/*.mp4
# do
#     python crop-video.py --inp $file
# done

driving_dir="asset/young"

driving_list="
0_F.mp4_crop.mp4
0_R.mp4_crop.mp4
45_F.mp4_crop.mp4
45_R.mp4_crop.mp4
90_-90_N.mp4_crop.mp4
90_F.mp4_crop.mp4
90_R.mp4_crop.mp4
"

source_list="
asset/kkj,30_30p.png_crop.png
asset/kkj,frame_reference.png_crop.png
asset/son,frame_30.png_crop.png
asset/son,frame_reference.png_crop.png
"

for i in $source_list
do 
    IFS=','
    set $i
    source_dir=$1
    source=$2
    echo "working on $source_dir/$source"
    IFS=$'\n'
    for j in $driving_list
    do
        set $j
        driving=$1
        echo "working on $driving_dir/$driving"
        python demo.py --config config/vox-256.yaml --checkpoint ../fv2v/ckpt/00000189-checkpoint.pth.tar --source_image "$source_dir/$source" --driving_video "$driving_dir/$driving" --relative --adapt_scale --find_best_frame --result_video "result/$source$driving"
    done
done