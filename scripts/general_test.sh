#!/bin/bash

. /home/server25/anaconda3/etc/profile.d/conda.sh

input_dir=result/video
checkpoint="/home/server25/minyeong_workspace/BFv2v/log/rotation_ckpt/00002849-checkpoint.pth.tar"
checkpoint_headmodel=log_headmodel/1.0_1.0/best.tar
source_dir=asset/celeb
driving_dir=asset/young_v2

i="1.0"
j="1.0"

conda deactivate

for source_image in $source_dir/*
do 
    for driving_video in $driving_dir/*
    do  
        conda activate fom
        vid=$(basename $source_image .png)_$(basename $driving_video .mp4).mp4
        echo "working on source image ${source_image}"
        echo "working on driving video ${driving_video}"
        echo "working on ${vid}"
        python drive_mesh.py --checkpoint "/home/server25/minyeong_workspace/BFv2v/log_headmodel/${i}_${j}/best.tar" --coef_e_tilda $i --coef_e_prime $j --src_img $source_image --drv_vid $driving_video
        python demo.py --config config/vox-256-renderer.yaml --checkpoint "${checkpoint}" --source_image $source_image --checkpoint_headmodel $checkpoint_headmodel --result_vid $vid --driven_dir log_headmodel/${i}_${j}
        conda deactivate
        ffmpeg -y -i ${input_dir}/${vid} -i $driving_video -filter_complex hstack=inputs=2 ${input_dir}/${vid}_compare.mp4
    done
done




