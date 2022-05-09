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

# Clone DeepStream repo
git clone https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps.git ${tao_triton_root}/deepstream_reference_apps
# git clone https://github.com/xunleiw/deepstream_reference_apps.git ${tao_triton_root}/deepstream_reference_apps
export BODYPOSE3D_HOME=${tao_triton_root}/deepstream_reference_apps/deepstream-bodypose-3d

# Download models using NGC
mkdir -p $BODYPOSE3D_HOME/models
cd $BODYPOSE3D_HOME/models
check_ngc_cli_installation
ngc registry model download-version ${pc_peoplenet_version}
ngc registry model download-version ${pc_bodypose3dnet_version}
apt-get install -y tree
tree $BODYPOSE3D_HOME -d

# Install Eigen
cd $BODYPOSE3D_HOME
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
tar xvzf eigen-3.4.0.tar.gz
ln eigen-3.4.0 eigen -s
rm eigen-3.4.0.tar.gz

# Update event message payload of DeepStream
cp $BODYPOSE3D_HOME/sources/deepstream-sdk/eventmsg_payload.cpp /opt/nvidia/deepstream/deepstream/sources/libs/nvmsgconv/deepstream_schema
apt-get install -y libjson-glib-dev uuid-dev
cd /opt/nvidia/deepstream/deepstream/sources/libs/nvmsgconv
make; make install

# Make 3D body pose sources
export CUDA_VER=${cuda_ver}
cd $BODYPOSE3D_HOME/sources/nvdsinfer_custom_impl_BodyPose3DNet
make
cd $BODYPOSE3D_HOME/sources
make

# Run 3D body pose
./deepstream-pose-estimation-app --input file://$BODYPOSE3D_HOME/streams/bodypose.mp4 \
                                 --output $BODYPOSE3D_HOME/streams/bodypose_3dbp.mp4 \
                                 --focal 800.0 \
                                 --width 1280 \
                                 --height 720 \
                                 --fps \
                                 --save-pose $BODYPOSE3D_HOME/streams/bodypose_3dbp.json

# Run the Triton client
cd ${tao_triton_root}
python -m tao_triton.python.entrypoints.tao_client $BODYPOSE3D_HOME/streams/bodypose_3dbp.json \
       --dataset_convert_config ${tao_triton_root}/tao_triton/python/dataset_convert_specs/dataset_convert_config_pose_classification.yaml \
       -m pose_classification_tao \
       -x 1 \
       -b 1 \
       --mode Pose_classification \
       -i https \
       -u localhost:8000 \
       --async \
       --output_path ${tao_triton_root}

# Clean repo
rm -r ${tao_triton_root}/deepstream_reference_apps
