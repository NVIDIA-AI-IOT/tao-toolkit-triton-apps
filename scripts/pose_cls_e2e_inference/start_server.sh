#!/bin/bash

function check_wget_installed {
    if ! command -v wget > /dev/null; then
        echo "Wget not found. Please run sudo apt-get install wget"
        return false
    fi
    return 0
}

function check_ngc_cli_installation {
    if ! command -v ngc > /dev/null; then
        echo "[ERROR] The NGC CLI tool not found on device in /usr/bin/ or PATH env var"
        echo "[ERROR] Please follow: https://ngc.nvidia.com/setup/installers/cli"
        exit
    fi
}

get_ngc_key_from_environment() {
    # first check the global NGC_API_KEY environment variable.
    local ngc_key=$NGC_API_KEY
    # if env variable was not set, and a ~/.ngc/config exists
    # try to get it from there.
    if [ -z "$ngc_key" ] && [[ -f "$HOME/.ngc/config" ]]
    then
        ngc_key=$(cat $HOME/.ngc/config | grep apikey -m1 | awk '{print $3}')
    fi
    echo $ngc_key
}

check_ngc_cli_installation
NGC_API_KEY="$(get_ngc_key_from_environment)"
if [ -z "$NGC_API_KEY" ]; then
    echo -e 'Did not find environment variable "$NGC_API_KEY"'
    read -sp 'Please enter API key for ngc.nvidia.com: ' NGC_API_KEY
    echo
fi

set -e

# Docker login to Nvidia GPU Cloud (NGC).
docker login nvcr.io -u \$oauthtoken -p ${NGC_API_KEY}

# Load config file
script_path="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
if [ -z "$1" ]; then
    config_path="${script_path}/../config.sh"
else
    config_path=$(readlink -f $1)
fi

# Configure the required environment variables.
if [[ ! -f $config_path ]]; then
    echo 'Unable to load configuration file. Override path to file with -c argument.'
    exit 1
fi
source $config_path

# Building the Triton docker with the tao-converter.
docker build -f "${tao_triton_root}/docker/Dockerfile" \
             -t ${tao_triton_server_docker}:${tao_triton_server_tag} ${tao_triton_root}

mkdir -p ${default_model_download_path} && cd ${default_model_download_path}
wget --content-disposition ${ngc_pose_classification} -O ${default_model_download_path}/poseclassificationnet_v1.0.zip && \
     unzip ${default_model_download_path}/poseclassificationnet_v1.0.zip -d ${default_model_download_path}/pose_cls_model/

# Run the server container.
echo "Running the server on ${gpu_id}"
find ${tao_triton_root}/model_repository -mindepth 1 ! -regex "^${tao_triton_root}/model_repository/pose_classification_tao\(/.*\)?" -delete
docker run -it --rm -v ${tao_triton_root}/model_repository:/model_repository \
           -v ${default_model_download_path}:/tao_models \
           -v ${tao_triton_root}/scripts:/tao_triton \
           --gpus all \
           -p 8000:8000 \
           -p 8001:8001 \
           -p 8002:8002 \
           -e CUDA_VISIBLE_DEVICES=$gpu_id \
           ${tao_triton_server_docker}:${tao_triton_server_tag} \
           /tao_triton/pose_cls_e2e_inference/download_and_convert.sh
