# TLT Triton Apps

- [TLT Triton Apps](#tlt-triton-apps)
  - [Installation Instructions](#installation-instructions)
    - [Pre-requisites](#pre-requisites)
    - [Install python dependencies](#install-python-dependencies)
    - [Running the client samples](#running-the-client-samples)

NVIDIA Transfer Learning Toolkit (TLT), provides users an easy interface to generate accurate and optimized models
for computer vision and conversational AI use cases. These models are generally deployed via the DeepStream SDK or
Jarvis pipelines.

This repository provides users with reference examples to infer the trained models with TLT in Triton. For this commit,
we provide reference applications for 2 computer vision models, namely:

- DetectNet_v2
- Image Classification

Triton is an NVIDIA developed inference software solution to efficiently deploy Deep Neural Networks (DNN) developed across several frameworks, for example TensorRT, Tensorflow, and ONNXRuntime. Triton Inference Server runs multiple models from the same or different frameworks concurrently on a single GPU. In a multi-GPU server, it automatically creates an instance of each model on each GPU. It also supports ensembling multiple models to build a pipeline.

The Triton inference architecture consists of 2 components

- Inference Server
- Triton Client

The *Inference Server* loads a model and spins up an inference context to which users can send inference requests to. The first step in loading a model is to serve your models using a model repository. This could be a file system, GCP, Asure or AWS s3. For the sake of this document, the model repository will be a local file system mounted on the server.

The *Triton Client* application is the user interface that sends inference requests to inference context spun up by the server. This can be written as a python client using the tritonclient package.

To understand Triton better, please refer to the official [documentation](https://github.com/triton-inference-server/server#readme).

## Installation Instructions

### Pre-requisites

In order to successfully run the examples defined in this repository, please install the following items.

| Component  | Version |
| :---  | :------ |
| python | 3.6.9 +  |
| python-pip | 21.06 |
| nvidia-pyindex| |
| virtualenvwrapper | |

### Install python dependencies

1. Set up virtualenvwrapper using the following instructions:

    You may follow the instructions in this here to setup a python virtualenv using a virtualenvwrapper.

    Once you have followed the instruction to install virtualenv and virtualenvwrapper, set the Python version in the virtualenv. This can be done in either of the following ways, by:

    Defining the environment variable called VIRTUALENVWRAPPER_PYTHON. This variable should point to the path where the python3 binary is installed in your local machine. You can also add it to your .bashrc or .bash_profile for setting your Python virtualenv by default.

    ```sh
    export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
    ```

    Setting the path to the python3 binary when creating your virtualenv using the virtualenv wrapper

    ```sh
    mkvirtualenv triton_dev -p /path/to/your/python3
    ```

    Once you have created this virtualenv, you may reinstantiate this virtualenv on any terminal session simply by running

    ```sh
    workon triton_dev
    ```

2. Install python-pip dependencies

    This repositories relies on several third party python dependancies, which you may install to your virtualenv using
    the following command.

    ```sh
    pip install -r requirements-pip.txt
    ```

3. Add the tlt_triton repository to the PYTHONPATH of the python environment.

    For a virtualenv, you may do so by executing the following command.

    ```sh
    add2virtualenv $TLT_TRITON_REPO_ROOT
    ```

    For native python, please run

    ```sh
    export PYTHONPATH=${TLT_TRITON_REPO_ROOT}:${PYTHONPATH}
    ```

### Running the client samples

The Triton client to serve run TLT models is implemented in the `tlt_triton/python/entrypoints/tlt_client.py`. This implementation is a reference
example run to `detectnet_v2` and `classification`.

The CLI options for this client application are as follows:

```sh
usage: tlt_client.py [-h] [-v] [-a] [--streaming] -m MODEL_NAME
                     [-x MODEL_VERSION] [-b BATCH_SIZE] [-c CLASSES]
                     [--mode {NONE,DetectNet_v2}] [-u URL] [-i PROTOCOL]
                     [--class_list CLASS_LIST] --output_path OUTPUT_PATH
                     --postprocessing_config POSTPROCESSING_CONFIG
                     [image_filename]

positional arguments:
  image_filename        Input image / Input folder.

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         Enable verbose output
  -a, --async           Use asynchronous inference API
  --streaming           Use streaming inference API. The flag is only
                        available with gRPC protocol.
  -m MODEL_NAME, --model-name MODEL_NAME
                        Name of model
  -x MODEL_VERSION, --model-version MODEL_VERSION
                        Version of model. Default is to use latest version.
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size. Default is 1.
  -c CLASSES, --classes CLASSES
                        Number of class results to report. Default is 1.
  --mode {NONE,DetectNet_v2}
                        Type of scaling to apply to image pixels. Default is
                        NONE.
  -u URL, --url URL     Inference server URL. Default is localhost:8000.
  -i PROTOCOL, --protocol PROTOCOL
                        Protocol (HTTP/gRPC) used to communicate with the
                        inference service. Default is HTTP.
  --class_list CLASS_LIST
                        Comma separated class names
  --output_path OUTPUT_PATH
                        Path to where the inferenced outputs are stored.
  --postprocessing_config POSTPROCESSING_CONFIG
                        Path to the DetectNet_v2 clustering config.
```

Assuming that a Triton inference server with a valid DNv2 engine has been set up, you
may run the inference sample by using the following command.

```sh
python tlt_client.py \
       /home/projects2_metropolis/datasets/kpi_datasets/People/DSC_0016E/images \
       -m peoplenet_tlt \
       -x 1 \
       -b 8 \
       --mode DetectNet_v2 \
       -i https \
       -u localhost:8000 \
       --async \
       --output_path /home/scratch.p3/vpraveen/experiments/triton_output_dsc0016_output_demo \
       --postprocessing_config /home/vpraveen/Software/triton/tlt_triton/python/clustering_specs/clustering_config.prototxt 
```
