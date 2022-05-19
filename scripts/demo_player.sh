#!/bin/bash

. /home/server25/anaconda3/etc/profile.d/conda.sh

input_dir=result/video/
out_dir=result/gpen/
baseline_dir=result/test_baseline
vid=E_00001_00004.mp4
checkpoint="/home/server25/minyeong_workspace/BFv2v/ckpt/00000019-checkpoint.pth.tar"
checkpoint_headmodel=log_headmodel/v4.2/best.tar

src_dir=asset/player
drv_dir=asset/obama
source_image=asset/celeb/E_00001_cropped.png
driving_video=asset/young_v2/00004_cropped.mp4
i="1.0"
j="1.0"

source_list="
E_00001
E_00002
W_00001
W_00002
W_00003
W_00006
"


driving_list="
00000
"

ckpt_list="
29
"

conda deactivate


for ckpt in $ckpt_list
    do
        checkpoint="/home/server25/minyeong_workspace/BFv2v/ckpt/000000${ckpt}-checkpoint.pth.tar"
        mkdir -p ${input_dir}/${ckpt}
        for src_path in $src_dir/*
        do
            if grep -q GPEN <<< $src_path;  then
                continue
            fi
            for drv in $driving_list
            do
                src=$(basename $src_path .png)
                source_image=asset/player/${src}_GPEN.png
                driving_video=${drv_dir}/${drv}_cropped.mp4
                vid=${src}_${drv}.mp4
                conda activate fom
                # python drive_mesh.py --checkpoint $checkpoint_headmodel --coef_e_tilda $i --coef_e_prime $j --src_img $source_image --drv_vid $driving_video

                python demo.py --config config/vox-256-renderer_v3.yaml --checkpoint "${checkpoint}" --source_image $source_image --checkpoint_headmodel $checkpoint_headmodel --result_dir $input_dir --result_vid $vid --driven_dir log_headmodel/v4.2 --driving_video $driving_video

                conda deactivate
                # mkdir -p ${out_dir}/$vid
                # ffmpeg -y -i $driving_video -i ${input_dir}/${vid} -filter_complex hstack=inputs=3 ${input_dir}/${ckpt}/${vid}_sound.mp4
                # ffmpeg -y -i ${input_dir}/${vid} -r 25 ${out_dir}/${vid}/%05d.png
                # # ffmpeg -y -i ${input_dir}/${vid} -vf scale=512:512 -r 25 ${out_dir}/${vid}/%05d.png

                # conda activate gpen
                # cd ../GPEN
                # python demo.py --task FaceEnhancement --model GPEN-BFR-256 --in_size 256 --channel_multiplier 1 --narrow 0.5 --use_sr --sr_scale 4 --use_cuda --save_face --indir ../BFv2v/${out_dir}/${vid} --outdir ../BFv2v/${out_dir}/${vid}_SR
                # cd ../BFv2v
                # conda deactivate
                # ffmpeg -y -r 25 -i ${out_dir}/${vid}_SR/%05d_GPEN.jpg -vf fps=25 ${out_dir}/${vid}_SR.mp4
                # ffmpeg -y -i ${out_dir}/${vid}_SR.mp4 -i $driving_video -map 0:v:0 -map 1:a:0 ${out_dir}/${vid}_SR_sound.mp4

            done
        done
    done
done


