#!/bin/bash

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

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
