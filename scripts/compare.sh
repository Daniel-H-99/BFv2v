#!/bin/bash

. /home/server25/anaconda3/etc/profile.d/conda.sh

# for file in /home/server25/minyeong_workspace/YOUNG/cropped/*.mp4
# do
#     python crop-video.py --inp $file
# done

driving_dir="asset/young_v2"

driving_list="
00010_cropped.mp4
"

source_list="
asset,justin,frame_reference.png
"

conda deactivate

for i in $source_list
do 
    IFS=','
    set $i
    source_dir=$1
    source_id=$2
    source=$3
    echo "working on $source_dir/$source_id/$source"
    if [ ! -d $source_dir/$source_id ]; then
        mkdir $source_dir/$source_id
    fi
    IFS=$'\n'
    for j in $driving_list
    do
        set $j
        driving=$1
        echo "working on $driving_dir/$driving"
        conda activate fom
        python demo.py --config config/vox-256.yaml --checkpoint _ckpt/00000299-checkpoint.pth.tar --source_image "$source_dir/$source_id/$source" --driving_video "$driving_dir/$driving" --relative --adapt_scale --find_best_frame --result_dir result --result_video "$source_id"_"$source"_"$driving"
        conda deactivate
        ffmpeg -y -i $source_dir/$source_id/$source -i $driving_dir/$driving -i result/video/"$source_id"_"$source"_"$driving" -filter_complex \
            "[0:v]scale=-1:256[v1]; \
            [1:v]scale=-1:256,crop=trunc(iw/2)*2:trunc(ih/2)*2[v2];\
            [2:v]scale=-1:256,crop=trunc(iw/2)*2:trunc(ih/2)*2[v3]; \
            [v1][v2][v3]hstack=inputs=3[v]" -map "[v]" result/compare/compare_"$source_id"_"$source"_"$driving"
    done
done
