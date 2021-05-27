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

tlt_triton_root=$PWD
gpu_id=0
tlt_triton_server_docker="nvcr.io/nvidia/tlt-triton"
tlt_triton_server_tag="21.03-py3"

tlt_key_peoplenet="tlt_encode"
tlt_key_dashcamnet="tlt_encode"
tlt_key_vehicletypenet="tlt_encode"

ngc_peoplenet="https://api.ngc.nvidia.com/v2/models/nvidia/tlt_peoplenet/versions/pruned_v2.1/zip"
ngc_dashcamnet="https://api.ngc.nvidia.com/v2/models/nvidia/tlt_dashcamnet/versions/pruned_v1.0/zip"
ngc_vehicletypenet="https://api.ngc.nvidia.com/v2/models/nvidia/tlt_vehicletypenet/versions/pruned_v1.0/zip"

default_model_download_path="${tlt_triton_root}/tlt_models"
