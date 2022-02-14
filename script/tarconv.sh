#!/bin/bash
source activate pt_env
cd /home/koh/work/2021/SC-CFR/
gpu=0
outcomes=('MEAN' 'STD')


for outcome in "${outcomes[@]}"
do
for i in {0..9} ; do
    export CUDA_VISIBLE_DEVICES=$(expr $gpu % 4)
    echo ${CUDA_VISIBLE_DEVICES}
    gpu=$(expr $gpu + 1)
    python demo_tarconv.py --expid ${i} --outcome ${outcome} &
    sleep 4
done
done