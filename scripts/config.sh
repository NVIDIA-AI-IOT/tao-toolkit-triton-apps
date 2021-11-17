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
tao_triton_server_docker="nvcr.io/nvidia/tao/triton-apps"
tao_triton_server_tag="21.11-py3"

# Load key for the models.
tlt_key_peoplenet="tlt_encode"
tlt_key_dashcamnet="tlt_encode"
tlt_key_vehicletypenet="tlt_encode"

# Setting model version to run inference on.
peoplenet_version="pruned_quantized_v2.1.1"
dashcamnet_version="pruned_v1.0.1"
vehicletypenet_version="pruned_v1.0.1"

# NGC URL's to download the model.
ngc_peoplenet="https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet/versions/${peoplenet_version}/zip"
ngc_dashcamnet="https://api.ngc.nvidia.com/v2/models/nvidia/tao/dashcamnet/versions/${dashcamnet_version}/zip"
ngc_vehicletypenet="https://api.ngc.nvidia.com/v2/models/nvidia/tao/vehicletypenet/versions/${vehicletypenet_version}/zip"

default_model_download_path="${tao_triton_root}/tao_models"
