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

# load config file
script_path="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
if [ -z "$1" ]; then
    config_path="${script_path}/config.sh"
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
wget --content-disposition ${ngc_peoplenet} -O ${default_model_download_path}/peoplenet_${peoplenet_version}.zip && \
     unzip ${default_model_download_path}/peoplenet_${peoplenet_version}.zip -d ${default_model_download_path}/peoplenet_model/
wget --content-disposition ${ngc_dashcamnet} -O ${default_model_download_path}/dashcamnet_${dashcamnet_version}.zip && \
     unzip ${default_model_download_path}/dashcamnet_${dashcamnet_version}.zip -d ${default_model_download_path}/dashcamnet_model/
wget --content-disposition ${ngc_vehicletypenet} -O ${default_model_download_path}/vehicletypenet_${vehicletypenet_version}.zip && \
     unzip ${default_model_download_path}/vehicletypenet_${vehicletypenet_version}.zip -d ${default_model_download_path}/vehicletypenet_model/
wget --content-disposition ${ngc_lprnet} -O ${default_model_download_path}/lprnet_pruned_v1.0.zip && \
     unzip ${default_model_download_path}/lprnet_pruned_v1.0.zip -d ${default_model_download_path}/lprnet_model/
wget --content-disposition ${ngc_yolov3} -O ${default_model_download_path}/models.zip && \
     unzip ${default_model_download_path}/models.zip -d ${default_model_download_path}  && \
     rm -rf ${default_model_download_path}/yolov3_model && \
     mv ${default_model_download_path}/models/yolov3  ${default_model_download_path}/yolov3_model && \
     rm -rf ${default_model_download_path}/retinanet_model && \
     mv ${default_model_download_path}/models/retinanet  ${default_model_download_path}/retinanet_model && \
     rm -rf ${default_model_download_path}/models
wget --content-disposition ${ngc_peoplesegnet}  -O ${default_model_download_path}/peoplesegnet_deployable_v2.0.zip  && \
     unzip ${default_model_download_path}/peoplesegnet_deployable_v2.0.zip -d ${default_model_download_path}/peoplesegnet_model/
rm -rf ${default_model_download_path}/multitask_cls_model
mkdir ${default_model_download_path}/multitask_cls_model
wget --no-check-certificate ${ngc_mcls_classification} -O ${default_model_download_path}/multitask_cls_model/multitask_cls_resnet18.etlt
wget --content-disposition ${ngc_pose_classification} -O ${default_model_download_path}/poseclassificationnet_v1.0.zip && \
     unzip ${default_model_download_path}/poseclassificationnet_v1.0.zip -d ${default_model_download_path}/pose_cls_model/
rm -rf ${default_model_download_path}/re_id_model
mkdir ${default_model_download_path}/re_id_model
wget --no-check-certificate ${ngc_re_identification} -O ${default_model_download_path}/re_id_model/resnet50_market1501.etlt
ngc registry model download-version $ngc_visual_changenet --dest ${default_model_download_path}
mv ${default_model_download_path}/visual_changenet_segmentation_levircd_v${visual_changenet_version} ${default_model_download_path}/visual_changenet_segmentation_tao
wget --content-disposition ${ngc_centerpose} -O ${default_model_download_path}/centerpose_ros_deployable_bottle_dla34_v1.0.zip && \
     unzip ${default_model_download_path}/centerpose_ros_deployable_bottle_dla34_v1.0.zip -d ${default_model_download_path}/centerpose_model/
rm -rf ${default_model_download_path}/*.zip
# wget --content-disposition ${ngc_foundationpose} -O ${default_model_download_path}/foundationpose_deployable.zip && \
#      unzip ${default_model_download_path}/foundationpose_deployable.0.zip -d ${default_model_download_path}/foundationpose_model/
# rm -rf ${default_model_download_path}/*.zip

# Run the server container.
echo "Running the server on ${gpu_id}"
docker run -it --rm -v ${tao_triton_root}/model_repository:/model_repository \
	        -v ${default_model_download_path}:/tao_models \
		    -v ${tao_triton_root}/scripts:/tao_triton \
		    --gpus all \
		    -p 8000:8000 \
		    -p 8001:8001 \
		    -p 8002:8002 \
		    -e CUDA_VISIBLE_DEVICES=$gpu_id \
		    ${tao_triton_server_docker}:${tao_triton_server_tag} \
		    /tao_triton/download_and_convert.sh
