#!/bin/bash -e
# From the tensorflow/models/research/ directory
PIPELINE_CONFIG_PATH=/media/jay/data/Dataset/object_detection/models/ssd_mobilenet_v1_ppn/pipeline.config
MODEL_DIR=/media/jay/data/Dataset/object_detection/models/model
NUM_TRAIN_STEPS=10000
NUM_EVAL_STEPS=500
python object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --num_eval_steps=${NUM_EVAL_STEPS} \
    --alsologtostderr
