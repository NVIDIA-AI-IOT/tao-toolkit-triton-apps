#!/bin/bash

# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
# 
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
# 
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

tao_triton_root=$PWD
gpu_id=0
cuda_ver=12.0
tao_triton_server_docker="nvcr.io/nvidia/tao/triton-apps"
tao_triton_server_tag="23.02-py3"

# Load key for the models.
tlt_key_peoplenet="tlt_encode"
tlt_key_dashcamnet="tlt_encode"
tlt_key_vehicletypenet="tlt_encode"
tlt_key_lprnet="nvidia_tlt"
tlt_key_yolov3="nvidia_tlt"
tlt_key_peoplesegnet="nvidia_tlt"
tlt_key_retinanet="nvidia_tlt"
tlt_key_multitask_classification="nvidia_tlt"
tlt_key_pose_classification="nvidia_tao"
tlt_key_re_identification="nvidia_tao"

# Setting model version to run inference on.
peoplenet_version="pruned_quantized_decrypted_v2.3.3"
dashcamnet_version="pruned_v1.0.4"
vehicletypenet_version="pruned_v1.0.1"
visual_changenet_version="visual_changenet_levircd_deployable_v1.0"
centerpose_version="deployable_bottle_fan_small_v1.0"
foundationpose_version="deployable_v1.0"

# Setting model version to run inference on for Pose Classification.
pc_peoplenet_version="nvidia/tao/peoplenet:deployable_quantized_v2.5"
pc_bodypose3dnet_version="nvidia/tao/bodypose3dnet:deployable_accuracy_v1.0"

# NGC URL's to download the model.
ngc_peoplenet="https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet/versions/${peoplenet_version}/zip"
ngc_dashcamnet="https://api.ngc.nvidia.com/v2/models/nvidia/tao/dashcamnet/versions/${dashcamnet_version}/zip"
ngc_vehicletypenet="https://api.ngc.nvidia.com/v2/models/nvidia/tao/vehicletypenet/versions/${vehicletypenet_version}/zip"
ngc_lprnet="https://api.ngc.nvidia.com/v2/models/nvidia/tao/lprnet/versions/deployable_v1.0/zip"
ngc_yolov3="https://nvidia.box.com/shared/static/3a00fdf8e1s2k3nezoxmfyykydxiyxy7"
ngc_peoplesegnet="https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplesegnet/versions/deployable_v2.0/zip"
ngc_retinanet="https://nvidia.box.com/shared/static/3a00fdf8e1s2k3nezoxmfyykydxiyxy7"
ngc_mcls_classification="https://docs.google.com/uc?export=download&id=1blJQDQSlLPU6zX3yRmXODRwkcss6B3a3"
ngc_pose_classification="https://api.ngc.nvidia.com/v2/models/nvidia/tao/poseclassificationnet/versions/deployable_v1.0/zip"
ngc_re_identification="https://drive.google.com/uc?export=download&id=1jicWzrPgEgvHLoxS57XLwk3o2xRbXeN_"
ngc_visual_changenet="nvidia/tao/visual_changenet_segmentation_levircd:${visual_changenet_version}"
ngc_centerpose="https://api.ngc.nvidia.com/v2/models/nvidia/tao/centerpose_ros/versions/${centerpose_version}/zip"
ngc_foundationpose="nvstaging/tao/foundation_pose:${foundationpose_version}"

default_model_download_path="${tao_triton_root}/tao_models"
