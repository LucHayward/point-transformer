#!/bin/sh

export PYTHONPATH=./
eval "$(conda shell.bash hook)"
conda activate pt
PYTHON=python

TRAIN_CODE=train.py
TEST_CODE=test.py

dataset=$1
exp_name=$2
exp_dir=exp/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}.yaml

mkdir -p ${model_dir} ${result_dir}
mkdir -p ${result_dir}/last
mkdir -p ${result_dir}/best
cp tool/train.sh tool/${TRAIN_CODE} ${config} tool/test.sh tool/${TEST_CODE} ${exp_dir}

now=$(date +"%Y%m%d_%H%M%S")
#First train the pretrained model
$PYTHON ${exp_dir}/${TRAIN_CODE} \
  --config=${config} \
  data_root $3 \
  save_path ${exp_dir} \
  weight /home/eco02/Luc/point-transformer/s3dis_xyz_2_class_head.pth \
#  resume /home/eco02/Luc/point-transformer/${model_dir}/model_best.pth \
  2>&1 | tee ${exp_dir}/train-$now.log


mv ${exp_dir}/model ${exp_dir}/model_pretrained
mkdir ${exp_dir}/model

#Next do the untrained model
$PYTHON ${exp_dir}/${TRAIN_CODE} \
  --config=${config} \
  data_root $3 \
  save_path ${exp_dir} \
#  resume /home/eco02/Luc/point-transformer/${model_dir}/model_best.pth \
  2>&1 | tee ${exp_dir}/train-$now.log
