#!/bin/bash

python training_pipelines/training_pipeline_torch.py ResNet50 --pretrained --augment --learning-rate 1e-4 --batch-size 64 --epochs 50 --root-dir ./data/dataset_split --save-weights-dir ./weights/resnet-50 --save-weights-freq 2
