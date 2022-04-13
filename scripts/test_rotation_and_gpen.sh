#!/bin/bash

. /home/server25/anaconda3/etc/profile.d/conda.sh

# for file in /home/server25/minyeong_workspace/YOUNG/cropped/*.mp4
# do
#     python crop-video.py --inp $file
# done

input_dir=result/video
out_dir=result/gpen/
vid=justin_0_F.mp4
checkpoint="/home/server25/minyeong_workspace/BFv2v/log/rotation_ckpt/00001399-checkpoint.pth.tar"
checkpoint_headmodel=log_headmodel/1.0_1.0/best.tar
source_image=asset/justin/frame_reference.png
driving_video=asset/young/0_F.mp4_crop.mp4

conda deactivate
conda activate fom
python demo.py --config config/vox-256-renderer.yaml --checkpoint "${checkpoint}" --source_image $source_image --driving_video $driving_video --checkpoint_headmodel $checkpoint_headmodel --result_vid $vid --ignore_emotion


conda deactivate
mkdir -p ${out_dir}/$vid
ffmpeg -y -i ${input_dir}/${vid} -vf scale=512:512 -r 25 ${out_dir}/${vid}/%05d.png

conda activate gpen
cd ../GPEN
python demo.py --task FaceEnhancement --model GPEN-BFR-512 --in_size 512 --channel_multiplier 2 --narrow 1 --use_sr --sr_scale 2 --use_cuda --save_face --indir ../BFv2v/${out_dir}/${vid} --outdir ../BFv2v/${out_dir}/${vid}_SR

cd ../BFv2v
conda deactivate
ffmpeg -y -r 25 -i ${out_dir}/${vid}_SR/%05d_GPEN.jpg -vf fps=25 ${out_dir}/${vid}_SR.mp4