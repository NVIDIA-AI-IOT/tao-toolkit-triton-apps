#!/bin/bash

echo "Converting the PeopleNet model"
mkdir -p /model_repository/nota_tao/1
tao-converter /tao_models/nota_model/tlt_bongmyeong_model.etlt \
              -k tlt_encode \
              -d 3,544,960 \
              -o output_cov/Sigmoid,output_bbox/BiasAdd \
              -m 16 \
              -e /model_repository/nota_tao/1/model.plan


# # Generate a peoplenet model.
# echo "Converting the PeopleNet model"
# mkdir -p /model_repository/peoplenet_tao/1
# tao-converter /tao_models/peoplenet_model/resnet34_peoplenet_pruned_int8.etlt \
#               -c /tao_models/peoplenet_model/resnet34_peoplenet_pruned_int8_dla.txt \
#               -k tlt_encode \
#               -d 3,544,960 \
#               -o output_cov/Sigmoid,output_bbox/BiasAdd \
#               -t int8 \
#               -m 16 \
#               -e /model_repository/peoplenet_tao/1/model.plan

# # Generate a dashcamnet model.
# echo "Converting the DashcamNet model"
# mkdir -p /model_repository/dashcamnet_tao/1
# tao-converter /tao_models/dashcamnet_model/resnet18_dashcamnet_pruned.etlt \
#               -k tlt_encode \
#               -c /tao_models/dashcamnet_model/dashcamnet_int8.txt \
#               -d 3,544,960 \
#               -o output_cov/Sigmoid,output_bbox/BiasAdd \
#               -t int8 \
#               -m 16 \
#               -e /model_repository/dashcamnet_tao/1/model.plan

# # Generate a vehicletypnet model.
# echo "Converting the VehicleTypeNet model"
# mkdir -p /model_repository/vehicletypenet_tao/1
# tao-converter /tao_models/vehicletypenet_model/resnet18_vehicletypenet_pruned.etlt \
#               -k tlt_encode \
#               -c /tao_models/vehicletypenet_model/vehicletypenet_int8.txt  \
#               -d 3,224,224 \
#               -o predictions/Softmax \
#               -t int8 \
#               -m 16 \
#               -e /model_repository/vehicletypenet_tao/1/model.plan

# # Generate an LPRnet model.
# echo "Converting the LPRNet model"
# mkdir -p /model_repository/lprnet_tao/1
# tao-converter /tao_models/lprnet_model/us_lprnet_baseline18_deployable.etlt \
#               -k nvidia_tlt \
#               -p image_input,1x3x48x96,4x3x48x96,16x3x48x96 \
#               -t fp16 \
#               -e /model_repository/lprnet_tao/1/model.plan

# # Generate a YOLOv3 model.
# echo "Converting the YOLOv3 model"
# mkdir -p /model_repository/yolov3_tao/1
# tao-converter /tao_models/yolov3_model/yolov3_resnet18.etlt \
#               -k nvidia_tlt \
#               -p Input,1x3x544x960,4x3x544x960,16x3x544x960 \
#               -o BatchedNMS \
#               -t fp16 \
#               -e /model_repository/yolov3_tao/1/model.plan

# # Generate a peoplesegnet model.
# echo "Converting the peoplesegnet model"
# mkdir -p /model_repository/peoplesegnet_tao/1
# tao-converter /tao_models/peoplesegnet_model/peoplesegnet_resnet50.etlt \
#               -k nvidia_tlt \
#               -d 3,576,960 \
#               -p Input,1x3x576x960,4x3x576x960,16x3x576x960 \
#               -o generate_detections,mask_fcn_logits/BiasAdd \
#               -t fp16 \
#               -e /model_repository/peoplesegnet_tao/1/model.plan

# # Generate a retinanet model.
# echo "Converting the Retinanet model"
# mkdir -p /model_repository/retinanet_tao/1
# tao-converter /tao_models/retinanet_model/retinanet_resnet18_epoch_080_its_trt8.etlt \
#               -k nvidia_tlt \
#               -d 3,544,960 \
#               -o NMS \
#               -t fp16 \
#               -e /model_repository/retinanet_tao/1/model.plan

# # Generate a multitask_classification model.
# echo "Converting the multitask_classification model"
# mkdir -p /model_repository/multitask_classification_tao/1
# tao-converter /tao_models/multitask_cls_model/multitask_cls_resnet18.etlt \
#               -k nvidia_tlt \
#               -d 3,80,60 \
#               -o base_color/Softmax,category/Softmax,season/Softmax \
#               -t fp16 \
#               -m 16 \
#               -e /model_repository/multitask_classification_tao/1/model.plan

/opt/tritonserver/bin/tritonserver --model-store /model_repository
