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

for i in $coef_e_tilda
do 
    for j in $coef_e_prime
    do
        echo "working on $i / $j"
        python extract_static.py --coef_e_tilda $i --coef_e_prime $j
    done
done