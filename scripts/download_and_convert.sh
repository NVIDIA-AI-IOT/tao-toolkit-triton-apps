#!/bin/bash

# Generate a peoplenet model.
echo "Converting the PeopleNet model"
mkdir -p /model_repository/peoplenet_tlt/1
tlt-converter /tlt_models/peoplenet_model/resnet34_peoplenet_pruned.etlt \
              -k tlt_encode \
              -d 3,544,960 \
              -o output_cov/Sigmoid,output_bbox/BiasAdd \
              -t fp32 \
              -m 16 \
              -e /model_repository/peoplenet_tlt/1/model.plan

# Generate a dashcamnet model.
echo "Converting the DashcamNet model"
mkdir -p /model_repository/dashcamnet_tlt/1
tlt-converter /tlt_models/dashcamnet_model/resnet18_dashcamnet_pruned.etlt \
              -k tlt_encode \
              -c /tlt_models/dashcamnet_model/dashcamnet_int8.txt \
              -d 3,544,960 \
              -o output_cov/Sigmoid,output_bbox/BiasAdd \
              -t int8 \
              -m 16 \
              -e /model_repository/dashcamnet_tlt/1/model.plan

# Generate a vehicletypnet model.
echo "Converting the VehicleTypeNet model"
mkdir -p /model_repository/vehicletypenet_tlt/1
tlt-converter /tlt_models/vehicletypenet_model/resnet18_vehicletypenet_pruned.etlt \
              -k tlt_encode \
              -c /tlt_models/vehicletypenet_model/vehicletypenet_int8.txt  \
              -d 3,224,224 \
              -o predictions/Softmax \
              -t int8 \
              -m 16 \
              -e /model_repository/vehicletypenet_tlt/1/model.plan

/opt/tritonserver/bin/tritonserver --model-store /model_repository