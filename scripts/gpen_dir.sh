#!/bin/bash
. /home/server25/anaconda3/etc/profile.d/conda.sh

# video=result/video/W_00001_00003.mp4

input_dir=/home/server25/minyeong_workspace/Experiment/data/demo/src_cropped
output_dir=/home/server25/minyeong_workspace/Experiment/data/demo/src
tmp_dir=/home/server25/minyeong_workspace/Experiment/data/demo/src_cropped_gpen

conda activate gpen
cd ../GPEN
python demo.py --task FaceEnhancement --model GPEN-BFR-256 --in_size 256 --channel_multiplier 1 --narrow 0.5 --use_sr --sr_scale 4 --use_cuda --save_face --indir $input_dir --outdir $tmp_dir

mv $tmp_dir/*_GPEN.png $output_dir

# rm -rf $tmp_dir
