#!/bin/bash

. /home/server25/anaconda3/etc/profile.d/conda.sh

# for file in /home/server25/minyeong_workspace/YOUNG/cropped/*.mp4
# do
#     python crop-video.py --inp $file
# done

coef_e_tilda="
1.0
1.5
2.0
"

coef_e_prime="
1.0
1.5
2.0
"
conda deactivate

for i in $coef_e_tilda
do 
    for j in $coef_e_prime
    do
        echo "working on $i / $j"
        conda activate fom
        python drive_mesh.py --checkpoint "/home/server25/minyeong_workspace/BFv2v/log_headmodel/${i}_${j}/best.tar" --coef_e_tilda $i --coef_e_prime $j --src_img asset/kkj/frame_reference.png --drv_vid asset/young/0_F.mp4
        conda deactivate
        ffmpeg -y -i log_headmodel/${i}_${j}/result/source.mp4 -i log_headmodel/${i}_${j}/result/driving.mp4 -i log_headmodel/${i}_${j}/result/driven.mp4 -filter_complex \
            "[0:v]scale=-1:256[v1]; \
            [1:v]scale=-1:256,crop=trunc(iw/2)*2:trunc(ih/2)*2[v2];\
            [2:v]scale=-1:256,crop=trunc(iw/2)*2:trunc(ih/2)*2[v3]; \
            [v1][v2][v3]hstack=inputs=3[v]" -map "[v]" log_headmodel/${i}_${j}/result/compare_${i}_${j}.mp4
    done
done