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

source_dir="/home/server25/minyeong_workspace/Experiment/qualitative/TPSMM/202205300237"
source_images="
W_00003
"

conda deactivate

ffmpeg -y -i "/home/server25/minyeong_workspace/Experiment/data/young_v2/00003_cropped.mp4" \
 -i ${source_dir}/00007_00003_cropped.mp4  \
 -i ${source_dir}/00008_00003_cropped.mp4  \
 -i ${source_dir}/00009_00003_cropped.mp4  \
 -i ${source_dir}/00010_00003_cropped.mp4  \
 -i ${source_dir}/00012_00003_cropped.mp4  \
 -filter_complex \
    "[0:v]scale=-1:256[v1]; \
    [1:v]scale=-1:256,crop=trunc(iw/2)*2:trunc(ih/2)*2[v2];\
    [2:v]scale=-1:256,crop=trunc(iw/2)*2:trunc(ih/2)*2[v3]; \
    [3:v]scale=-1:256,crop=trunc(iw/2)*2:trunc(ih/2)*2[v4]; \
    [4:v]scale=-1:256,crop=trunc(iw/2)*2:trunc(ih/2)*2[v5]; \
    [5:v]scale=-1:256,crop=trunc(iw/2)*2:trunc(ih/2)*2[v6]; \
    [v1][v2][v3][v4][v5][v6]hstack=inputs=6[v]" -map "[v]" ${source_dir}/demo_tpsmm.mp4
