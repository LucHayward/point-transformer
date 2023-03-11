#!/bin/bash

cd ~/Luc/point-transformer
for dataset in Church Lunnahoja Monument Bagni_Nerone Montelupo Piazza
do
for split in "2.5%" "5%" "25%" "50%"
  do
  # Make dataset_split/dataset_split_xyz.yaml
#  mkdir config/${dataset}_${split}/
#  cp config/base${split}.yaml config/${dataset}_${split}/${dataset}_${split}_xyz.yaml

  # Run the training (set a specific stop point)
#  bash tool/train_many.sh "${dataset}_${split}" xyz dataset/PatrickData/${dataset}/${split}
#  bash tool/train_many.sh "${dataset}_${split}" xyz dataset/PatrickData/${dataset}/50%

  # Run just the validation/testing on the normal 50% data as well


#  Clear the FUCKING /dev/shm/
  rm /dev/shm/*
  done
done