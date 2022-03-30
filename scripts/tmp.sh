#!/bin/bash

. /home/server25/anaconda3/etc/profile.d/conda.sh

# for file in /home/server25/minyeong_workspace/YOUNG/cropped/*.mp4
# do
#     python crop-video.py --inp $file
# done

input_dir=result/video
out_dir=result/gpen/
vid=result.mp4

conda deactivate

# mkdir -p ${out_dir}/$vid

# ffmpeg -y -i ${input_dir}/${vid} -vf scale=512:512 -r 25 ${out_dir}/${vid}/%05d.png

ffmpeg -y -r 25 -i ${out_dir}/result_sr.mp4/%05d_GPEN.jpg -vf fps=25 ${out_dir}/result_gpen.mp4