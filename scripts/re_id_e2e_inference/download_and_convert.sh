#!/bin/bash

# Generate a re_identification model.
echo "Converting the re_identification model"
mkdir -p /model_repository/re_identification_tao/1
tao-converter /tao_models/re_id_model/resnet50_market1501.etlt \
              -k nvidia_tao \
              -d 3,256,128 \
              -p input,1x3x256x128,4x3x256x128,16x3x256x128 \
              -o fc_pred \
              -t fp16 \
              -m 16 \
              -e /model_repository/re_identification_tao/1/model.plan
/opt/tritonserver/bin/tritonserver --model-store /model_repository