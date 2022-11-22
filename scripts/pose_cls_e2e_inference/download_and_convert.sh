#!/bin/bash

# Generate a pose_classification model.
echo "Converting the pose_classification model"
mkdir -p /model_repository/pose_classification_tao/1
tao-converter /tao_models/pose_cls_model/st-gcn_3dbp_nvidia.etlt \
              -k nvidia_tao \
              -d 3,300,34,1 \
              -p input,1x3x300x34x1,4x3x300x34x1,16x3x300x34x1 \
              -o fc_pred \
              -t fp16 \
              -m 16 \
              -e /model_repository/pose_classification_tao/1/model.plan

/opt/tritonserver/bin/tritonserver --model-store /model_repository
