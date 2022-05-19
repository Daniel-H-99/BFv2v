#!/bin/bash

. /home/server25/anaconda3/etc/profile.d/conda.sh

# for file in /home/server25/minyeong_workspace/YOUNG/cropped/*.mp4
# do
#     python crop-video.py --inp $file
# done

# src_img_list="
# son
# "
# i="1.0"
# j="1.0"
# conda deactivate

# for src_img in $src_img_list
# do
#     echo "working on $src_img"
#     conda activate fom
#     python drive_mesh.py --checkpoint "/home/server25/minyeong_workspace/BFv2v/log_headmodel/${i}_${j}/best.tar" --coef_e_tilda $i --coef_e_prime $j --src_img asset/$src_img/frame_ugly.png --drv_vid asset/young/0_F.mp4 --res_dir result/driven_mesh/${src_img}_ugly
# done

# source_dir="asset/celeb/"
# source_images="
# E_00002
# W_00001
# W_00002
# W_00003
# W_00004
# "

# conda deactivate

# ffmpeg -y -i result/tmp/E_00002_meshed.mp4 -i result/tmp/young_img_00000_00001_meshed.mp4 -i result/tmp/00001_meshed.mp4 -filter_complex \
#     "[0:v]scale=-1:256[v1]; \
#     [1:v]scale=-1:256,crop=trunc(iw/2)*2:trunc(ih/2)*2[v2];\
#     [2:v]scale=-1:256,crop=trunc(iw/2)*2:trunc(ih/2)*2[v3]; \
#     [v1][v2][v3]hstack=inputs=3[v]" -map "[v]" result/tmp/compare_young_img_00000_00001.mp4


num_pc_list="
20
40
60
80
"

source_dir=asset/celeb
driving_dir=asset/young_v2
threshold=0.00

conda deactivate

for num_pc in $num_pc_list
do
    conda activate fom
    label="v4.${num_pc}"
    checkpoint="log_headmodel/${label}"
    python extract_static.py --num_pc $num_pc --checkpoint $checkpoint

    for source_image in $source_dir/*
    do 
        for driving_video in $driving_dir/*
        do
            # source_image="asset/celeb/E_00002_cropped.png"
            driving_video="asset/young_v2/00007_cropped.mp4"

            echo "working on $source_image / $driving_video"
            vid=$(basename $source_image .png)_$(basename $driving_video .mp4).mp4
            conda activate fom
            python drive_mesh.py --checkpoint "${checkpoint}/00000009-checkpoint.pth.tar" --src_img "$source_image" --drv_vid "$driving_video" --threshold $threshold
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
    done  
done