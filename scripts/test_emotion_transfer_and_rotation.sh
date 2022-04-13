#!/bin/bash

. /home/server25/anaconda3/etc/profile.d/conda.sh

input_dir=result/video
out_dir=result/gpen/
vid=justin_00010.mp4
checkpoint="/home/server25/minyeong_workspace/BFv2v/log/rotation_ckpt/00002849-checkpoint.pth.tar"
checkpoint_headmodel=log_headmodel/1.0_1.0/best.tar
source_image=asset/justin/frame_reference.png
driving_video=asset/young_v2/00010_cropped.mp4
i="1.0"
j="1.0"

conda deactivate
conda activate fom

python drive_mesh.py --checkpoint "/home/server25/minyeong_workspace/BFv2v/log_headmodel/${i}_${j}/best.tar" --coef_e_tilda $i --coef_e_prime $j --src_img $source_image --drv_vid $driving_video

python demo.py --config config/vox-256-renderer.yaml --checkpoint "${checkpoint}" --source_image $source_image --checkpoint_headmodel $checkpoint_headmodel --result_vid $vid --driven_dir log_headmodel/${i}_${j}

conda deactivate
mkdir -p ${out_dir}/$vid
ffmpeg -y -i ${input_dir}/${vid} -i $driving_video -filter_complex hstack=inputs=2 ${input_dir}/${vid}_sound.mp4
# ffmpeg -y -i ${input_dir}/${vid} -vf scale=512:512 -r 25 ${out_dir}/${vid}/%05d.png

# conda activate gpen
# cd ../GPEN
# python demo.py --task FaceEnhancement --model GPEN-BFR-512 --in_size 512 --channel_multiplier 2 --narrow 1 --use_sr --sr_scale 2 --use_cuda --save_face --indir ../BFv2v/${out_dir}/${vid} --outdir ../BFv2v/${out_dir}/${vid}_SR

# cd ../BFv2v
# conda deactivate
# ffmpeg -y -r 25 -i ${out_dir}/${vid}_SR/%05d_GPEN.jpg -vf fps=25 ${out_dir}/${vid}_SR.mp4
# ffmpeg -y -i ${out_dir}/${vid}_SR.mp4 -i $driving_video -map 0:v:0 -map 1:a:0 ${out_dir}/${vid}_SR_sound.mp4

