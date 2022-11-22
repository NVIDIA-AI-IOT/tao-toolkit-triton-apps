#!/bin/bash

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

# Download the dataset
rm -rf ${tao_triton_root}/data
mkdir ${tao_triton_root}/data
pip3 install gdown
gdown https://drive.google.com/uc?id=0B8-rUzbwVRk0c054eEozWG9COHM -O ${tao_triton_root}/data/market1501.zip

# Extract the files
unzip ${tao_triton_root}/data/market1501.zip -d ${tao_triton_root}/data
mv ${tao_triton_root}/data/Market-1501-v15.09.15 ${tao_triton_root}/data/market1501
rm ${tao_triton_root}/data/market1501.zip

# Verify
ls -l ${tao_triton_root}/data/market1501

# Sample the dataset
cd ${tao_triton_root}
python3 ./scripts/re_id_e2e_inference/sample_dataset.py \
        ${tao_triton_root}/data/market1501

# Run the Triton client
rm -f ./scripts/re_id_e2e_inference/results.json
python3 -m tao_triton.python.entrypoints.tao_client ${tao_triton_root}/data/market1501/sample_query \
        --test_dir ${tao_triton_root}/data/market1501/sample_test \
        -m re_identification_tao \
        -x 1 \
        -b 16 \
        --mode Re_identification \
        -i https \
        -u localhost:8000 \
        --async \
        --output_path ${tao_triton_root}/scripts/re_id_e2e_inference

# Plot inference results
python3 ./scripts/re_id_e2e_inference/plot_e2e_inference.py \
        ./scripts/re_id_e2e_inference/results.json \
        ./scripts/re_id_e2e_inference
