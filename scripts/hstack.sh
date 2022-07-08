#!/bin/bash

. /root/anaconda3/etc/profile.d/conda.sh

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

source_dir="/root/workspace/Experiment/tmp/new"
source_images="
W_00003
"

# conda deactivate
conda activate fom2

ffmpeg -y -i ${source_dir}/source.png \
 -i ${source_dir}/driving.mp4  \
 -i ${source_dir}/ours.mp4  \
 -filter_complex \
    "[0:v]scale=-1:256[v1]; \
    [1:v]scale=-1:256,crop=trunc(iw/2)*2:trunc(ih/2)*2[v2];\
    [2:v]scale=-1:256,crop=trunc(iw/2)*2:trunc(ih/2)*2[v3];\
    [v1][v2][v3]hstack=inputs=3[v]" -map "[v]" -map 1:a:0 ${source_dir}/demo.mp4

