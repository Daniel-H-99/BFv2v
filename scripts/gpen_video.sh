#!/bin/bash
. /home/server25/anaconda3/etc/profile.d/conda.sh

video=result/video/W_00001_00003.mp4

video_name=$(basename $video .mp4)
video_dir=$(dirname $video)
cuts_dir=$video_dir/${video_name}_cuts

conda deactivate
rm -r $cuts_dir
mkdir $cuts_dir

ffmpeg -y -ss 00:00:00 -i $video -t 00:00:10 -vf scale=512:512 -r 25 $cuts_dir/%05d.png

conda activate gpen
cd ../GPEN
python demo.py --task FaceEnhancement --model GPEN-BFR-256 --in_size 256 --channel_multiplier 1 --narrow 0.5 --use_sr --sr_scale 4 --use_cuda --save_face --indir ../BFv2v/${cuts_dir} --outdir ../BFv2v/${cuts_dir}_SR

cd ../BFv2v
conda deactivate
ffmpeg -y -r 25 -i ${cuts_dir}_SR/%05d_GPEN.jpg -vf fps=25 $video_dir/${video_name}_SR.mp4
# ffmpeg -y -i $video_dir/${video_name}_SR.mp4 -i $video -map 0:v:0 -map 1:a:0 $video_dir/${video_name}_SR_sound.mp4

rm -r $cuts_dir