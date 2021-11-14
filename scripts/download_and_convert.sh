#!/bin/bash

# Generate a peoplenet model.
echo "Converting the PeopleNet model"
mkdir -p /model_repository/peoplenet_tao/1
tao-converter /tlt_models/peoplenet_model/resnet34_peoplenet_pruned.etlt \
              -k tlt_encode \
              -d 3,544,960 \
              -o output_cov/Sigmoid,output_bbox/BiasAdd \
              -t fp32 \
              -m 16 \
              -e /model_repository/peoplenet_tao/1/model.plan

# Generate a dashcamnet model.
echo "Converting the DashcamNet model"
mkdir -p /model_repository/dashcamnet_tao/1
tao-converter /tlt_models/dashcamnet_model/resnet18_dashcamnet_pruned.etlt \
              -k tlt_encode \
              -c /tlt_models/dashcamnet_model/dashcamnet_int8.txt \
              -d 3,544,960 \
              -o output_cov/Sigmoid,output_bbox/BiasAdd \
              -t int8 \
              -m 16 \
              -e /model_repository/dashcamnet_tao/1/model.plan

# Generate a vehicletypnet model.
echo "Converting the VehicleTypeNet model"
mkdir -p /model_repository/vehicletypenet_tao/1
tao-converter /tlt_models/vehicletypenet_model/resnet18_vehicletypenet_pruned.etlt \
              -k tlt_encode \
              -c /tlt_models/vehicletypenet_model/vehicletypenet_int8.txt  \
              -d 3,224,224 \
              -o predictions/Softmax \
              -t int8 \
              -m 16 \
              -e /model_repository/vehicletypenet_tao/1/model.plan

# Generate a vehicletypnet model.
echo "Converting the LPRNet model"
mkdir -p /model_repository/lprnet_tao/1
tao-converter /tlt_models/lprnet_model/us_lprnet_baseline18_deployable.etlt \
              -k nvidia_tlt \
	      -p image_input,1x3x48x96,4x3x48x96,16x3x48x96 \
	      -t fp16 \
              -e /model_repository/lprnet_tao/1/model.plan  \


/opt/tritonserver/bin/tritonserver --model-store /model_repository
