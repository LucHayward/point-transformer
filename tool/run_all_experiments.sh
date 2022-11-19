#!/bin/bash

cd ~/Luc/point-transformer
for dataset in church Monument Montelupo Lunnahoja Piazza Bagni_Nerone
do
for split in "2.5%" "5%" "25%" "50%"
  do
  # Make dataset_split/dataset_split_xyz.yaml
  mkdir config/${dataset}_${split}/
  cp config/base${split}.yaml config/${dataset}_${split}/${dataset}_${split}_xyz.yaml

  # Run the training (set a specific stop point)
  bash tool/train_many.sh "${dataset}_${split}" xyz dataset/${dataset}/${split}

  # Run just the validation/testing on the normal 50% data as well
  done
done