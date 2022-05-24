#!/bin/bash

. /home/server25/anaconda3/etc/profile.d/conda.sh

# for file in /home/server25/minyeong_workspace/YOUNG/cropped/*.mp4
# do
#     python crop-video.py --inp $file
# done

coef_e_tilda="
1.0
"

coef_e_prime="
1.0
"

conda deactivate

label="v4.2"
checkpoint="log_headmodel/${label}"
source_dir=/home/server25/minyeong_workspace/Experiment/data/demo/src
driving_dir=/home/server25/minyeong_workspace/Experiment/data/demo/speech
threshold=0.00

for source_image in $source_dir/*
do 
    for driving_video in $driving_dir/*
    do
        # source_image="asset/celeb/E_00002_cropped.png"
        # driving_video="asset/young_v2/00007_cropped.mp4"

        echo "working on $source_image / $driving_video"
        vid=$(basename $source_image .png)_$(basename $driving_video .mp4).mp4
        conda activate fom
        python drive_mesh.py --checkpoint "${checkpoint}/best.tar" --src_img "$source_image" --drv_vid "$driving_video" --threshold $threshold
        conda deactivate
        mkdir -p result/"${label}"
        ffmpeg -y -i "$source_image" -i "${checkpoint}"/result/source_raw.mp4 -i "${checkpoint}"/result/source.mp4 -i "$driving_video" -i "${checkpoint}"/result/driving.mp4 -i "${checkpoint}"/result/driven.mp4 -filter_complex \
            "[0:v]scale=-1:256[v1]; \
            [1:v]scale=-1:256,crop=trunc(iw/2)*2:trunc(ih/2)*2[v2];\
            [2:v]scale=-1:256,crop=trunc(iw/2)*2:trunc(ih/2)*2[v3]; \
            [3:v]scale=-1:256,crop=trunc(iw/2)*2:trunc(ih/2)*2[v4]; \
            [4:v]scale=-1:256,crop=trunc(iw/2)*2:trunc(ih/2)*2[v5]; \
            [5:v]scale=-1:256,crop=trunc(iw/2)*2:trunc(ih/2)*2[v6]; \
            [v1][v2][v3][v4][v5][v6]hstack=inputs=6[v]" -map "[v]" result/${label}/$vid
        break
    done
    break
done  