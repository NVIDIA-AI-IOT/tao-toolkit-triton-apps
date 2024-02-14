# TAO Toolkit Triton Apps

- [Quick Start Instructions](#quick-start-instructions)
  - [Pre-requisites](#pre-requisites)
  - [Install python dependencies](#install-python-dependencies)
  - [Instantiate the Triton Server with sample models downloaded from NGC](#instantiate-the-triton-server-with-sample-models-downloaded-from-ngc)
  - [Running the client samples](#running-the-client-samples)
- [Configuring the TAO Toolkit client](docs/configuring_the_client.md#configuring-the-client-samples)
  - [DetectNet_v2](docs/configuring_the_client.md#detectnet-v2)
    - [Configuring the DetectNet_v2 model entry in the model repository](docs/configuring_the_client.md#configuring-the-detectnet-v2-model-entry-in-the-model-repository)
    - [Configuring the Post-processor](docs/configuring_the_client.md#configuring-the-post-processor)
  - [Classification](docs/configuring_the_client.md#classification)
    - [Configuring the Classification model entry in the model repository](docs/configuring_the_client.md#configuring-the-classification-model-entry-in-the-model-repository)
  - [LPRNet](docs/configuring_the_client.md#lprnet)
    - [Configuring the LPRNet model entry in the model repository](docs/configuring_the_client.md#configuring-the-lprnet-model-entry-in-the-model-repository)
    - [Configuring the LPRNet model Post-processor](docs/configuring_the_client.md#configuring-the-lprnet-model-post-processor)
  - [YOLOv3](docs/configuring_the_client.md#yolov3)
    - [Configuring the YOLOv3 model entry in the model repository](docs/configuring_the_client.md#configuring-the-yolov3-model-entry-in-the-model-repository)
    - [Configuring the YOLOv3 model Post-processor](docs/configuring_the_client.md#configuring-the-yolov3-model-post-processor)
  - [Peoplesegnet](docs/configuring_the_client.md#peoplesegnet)
    - [Configuring the Peoplesegnet model entry in the model repository](docs/configuring_the_client.md#configuring-the-peoplesegnet-model-entry-in-the-model-repository)
    - [Configuring the Peoplesegnet model Post-processor](docs/configuring_the_client.md#configuring-the-peoplesegnet-model-post-processor)
  - [Retinanet](docs/configuring_the_client.md#retinanet)
    - [Configuring the Retinanet model entry in the model repository](docs/configuring_the_client.md#configuring-the-retinanet-model-entry-in-the-model-repository)
    - [Configuring the Retinanet model Post-processor](docs/configuring_the_client.md#configuring-the-retinanet-model-post-processor)
  - [Multitask_classification](docs/configuring_the_client.md#multitask_classification)
    - [Configuring the Multitask_classification model entry in the model repository](docs/configuring_the_client.md#configuring-the-multitask_classification-model-entry-in-the-model-repository)
    - [Configuring the Multitask_classification model Post-processor](docs/configuring_the_client.md#configuring-the-multitask_classification-model-post-processor)
  - [Pose_classification](docs/configuring_the_client.md#pose_classification)
    - [Configuring the Pose_classification model entry in the model repository](docs/configuring_the_client.md#configuring-the-pose_classification-model-entry-in-the-model-repository)
    - [Configuring the Pose_classification model Post-processor](docs/configuring_the_client.md#configuring-the-pose_classification-model-post-processor)
    - [Configuring the Pose_classification data converter](docs/configuring_the_client.md#configuring-the-pose_classification-data-converter)
  - [Re_identification](docs/configuring_the_client.md#re_identification)
    - [Configuring the Re_identification model entry in the model repository](docs/configuring_the_client.md#configuring-the-re_identification-model-entry-in-the-model-repository)
    - [Configuring the Re_identification model Post-processor](docs/configuring_the_client.md#configuring-the-re_identification-model-post-processor)
  - [VisualChangeNet](docs/configuring_the_client.md#visual_changenet)
    - [Configuring the Re_identification model entry in the model repository](docs/configuring_the_client.md#configuring-the-visual-changenet-model-entry-in-the-model-repository)
    - [Configuring the Visual ChangeNet model Post-processor](docs/configuring_the_client.md#configuring-the-visual-changenet-model-post-processor)
  - [CenterPose](docs/configuring_the_client.md#centerpose)
    - [Configuring the CenterPose model entry in the model repository](docs/configuring_the_client.md#configuring-the-centerpose-model-entry-in-the-model-repository)
    - [Configuring the CenterPose model Post-processor](docs/configuring_the_client.md#configuring-the-centerpose-model-post-processor)

NVIDIA Train Adapt Optimize (TAO) Toolkit, provides users an easy interface to generate accurate and optimized models
for computer vision and conversational AI use cases. These models are generally deployed via the DeepStream SDK or
Jarvis pipelines.

This repository provides users with reference examples to infer the trained models with TAO Toolkit in Triton. For this commit,
we provide reference applications for 6 computer vision models and 1 character recognition model, namely:

- DetectNet_v2
- Image Classification
- LPRNet
- YOLOv3
- Peoplesegnet(Maskrcnn)
- Retinanet
- Multitask Classification
- Pose Classification
- Re-Identification
- VisualChangeNet
- CenterPose

Triton is an NVIDIA developed inference software solution to efficiently deploy Deep Neural Networks (DNN) developed
across several frameworks, for example TensorRT, Tensorflow, and ONNXRuntime. Triton Inference Server runs multiple
models from the same or different frameworks concurrently on a single GPU. In a multi-GPU server, it automatically
creates an instance of each model on each GPU. It also supports ensembling multiple models to build a pipeline.

The Triton inference architecture consists of 2 components

- Inference Server
- Triton Client

The *Inference Server* loads a model and spins up an inference context to which users can send inference requests to.
The first step in loading a model is to serve your models using a model repository. This could be a file system, GCP,
Asure or AWS s3. For the sake of this document, the [model repository](https://github.com/triton-inference-server/server/blob/main/docs/model_repository.md)
will be a local file system mounted on the server. Instructions on how to organize the layout of the model repository
such that it can be parsed by triton inference server are captured
[here](https://github.com/triton-inference-server/server/blob/main/docs/model_repository.md#repository-layout).

The *Triton Client* application is the user interface that sends inference requests to inference context
spun up by the server. This can be written as a python client using the tritonclient package.

To understand Triton better, please refer to the official [documentation](https://github.com/triton-inference-server/server#readme).

## Quick Start Instructions

Inorder to run the reference TAO Toolkit Triton client implementations in this TAO Toolkit, please follow the steps mentioned below:

### Pre-requisites

In order to successfully run the examples defined in this repository, please install the following items.

| **Component**  | **Version** |
| :---  | :------ |
| python | 3.6.9 +  |
| python3-pip | >19.03.5 |
| nvidia-container-toolkit | >1.3.0-1 |
| nvidia-driver | >455 |
| nvidia-pyindex| |
| virtualenvwrapper | |
| docker-ce | 20.10.6 |

### Install python dependencies

- Set up virtualenvwrapper using the following instructions:

  You may follow the instructions in this here to setup a python virtualenv using a virtualenvwrapper.

  Once you have followed the instruction to install virtualenv and virtualenvwrapper, set the Python version
  in the virtualenv. This can be done in either of the following ways, by:

  Defining the environment variable called VIRTUALENVWRAPPER_PYTHON. This variable should point to the path
  where the python3 binary is installed in your local machine. You can also add it to your .bashrc or .bash_profile
  for setting your Python virtualenv by default.

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

- Install python-pip dependencies

  This repositories relies on several third party python dependancies, which you may install to your virtualenv using
  the following command.

  ```sh
  pip3 install -r requirements-pip.txt
  ```

- Install the tritonclient library.

  The NVIDIA TritonClient library is hosted on the nvidia-pyindex repository. You may execute the following commands, to
  install it.

  ```sh
  pip3 install nvidia-pyindex
  pip3 install tritonclient[all]
  ```

- Add the tao_triton repository to the PYTHONPATH of the python environment.

  For a virtualenv, you may do so by executing the following command.

  ```sh
  add2virtualenv $TAO_TRITON_REPO_ROOT/tao_triton
  ```

  For native python, please run

  ```sh
  export PYTHONPATH=${TAO_TRITON_REPO_ROOT}/tao_triton:${PYTHONPATH}
  ```

### Instantiate the Triton Server with sample models downloaded from NGC

The Triton model client applications in the repository requires users to set-up a Triton server using a
TensorRT engine file. When running export, TAO Toolkit generates a `.etlt` file which is an intermediate format
that can moved across hardware platforms.

This sample walks through setting up instances of inferencing the following models

1. DashcamNet
2. PeopleNet
3. VehicleTypeNet
4. LPRNet
5. YOLOv3
6. Peoplesegnet
7. Retinanet
8. Multitask_classification
9. Pose_classification
10. Re_identification
11. VisualChangeNet
12. CenterPose

Simply run the quick start script:

 ```sh
 bash scripts/start_server.sh
 ```

### Running the client samples

The Triton client to serve run TAO Toolkit models is implemented in the `${TAO_TRITON_REPO_ROOT}/tao_triton/python/entrypoints/tao_client.py`.
This implementation is a reference example run to `detectnet_v2` , `classification` ,`LPRNet` , `YOLOv3` , `Peoplesegnet` , `Retinanet` , `Multitask_classification`, `Pose_classification`, `VisualChangeNet` and `CenterPose`.

The CLI options for this client application are as follows:

```text
usage: tao_client.py [-h] [-v] [-a] [--streaming] -m MODEL_NAME [-x MODEL_VERSION] [-b BATCH_SIZE]
                     [--mode {Classification,DetectNet_v2,LPRNet,YOLOv3,Peoplesegnet,Retinanet,Multitask_classification,Pose_classification,Re_identification,VisualChangeNet,CenterPose}]
                     [--img_dirs IMG_DIRS] [-u URL] [-i PROTOCOL] [--class_list CLASS_LIST] --output_path OUTPUT_PATH [--postprocessing_config POSTPROCESSING_CONFIG]
                     [--dataset_convert_config DATASET_CONVERT_CONFIG] [--test_dir TEST_DIR]
                     [image_filename]

positional arguments:
  image_filename        Input image / Input folder / Input JSON / Input pose sequences.

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         Enable verbose output
  -a, --async           Use asynchronous inference API
  --streaming           Use streaming inference API. The flag is only available with gRPC protocol.
  -m MODEL_NAME, --model-name MODEL_NAME
                        Name of model
  -x MODEL_VERSION, --model-version MODEL_VERSION
                        Version of model. Default is to use latest version.
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size. Default is 1.
  --mode {Classification,DetectNet_v2,LPRNet,YOLOv3,Peoplesegnet,Retinanet,Multitask_classification,Pose_classification,Re_identification,VisualChangeNet,CenterPose}
                        Type of scaling to apply to image pixels. Default is NONE.
  --img_dirs IMG_DIRS   Relative directory names for Siamese network input images
  -u URL, --url URL     Inference server URL. Default is localhost:8000.
  -i PROTOCOL, --protocol PROTOCOL
                        Protocol (HTTP/gRPC) used to communicate with the inference service. Default is HTTP.
  --class_list CLASS_LIST
                        Comma separated class names
  --output_path OUTPUT_PATH
                        Path to where the inferenced outputs are stored.
  --postprocessing_config POSTPROCESSING_CONFIG
                        Path to the DetectNet_v2 clustering config.
  --dataset_convert_config DATASET_CONVERT_CONFIG
                        Path to the Pose Classification dataset conversion config.
  --test_dir TEST_DIR   Path to the Re Identification test image directory.
```

Assuming that a Triton inference server with a valid Detectnet_v2 TensorRT engine has
been set up, you may run the inference sample by using the following command.

For example,

1. For PeopleNet:

  ```sh
  python tao_client.py \
        /path/to/a/directory/of/images \
        -m peoplenet_tao \
        -x 1 \
        -b 8 \
        --mode DetectNet_v2 \
        --class_list person,bag,face \
        -i https \
        -u localhost:8000 \
        --async \
        --output_path /path/to/the/output/directory \
        --postprocessing_config $tao_triton_root/tao_triton/python/clustering_specs/clustering_config_peoplenet.prototxt 
  ```

2. For DashCamNet:

  ```sh
  python tao_client.py \
        /path/to/a/directory/of/images \
        -m dashcamnet_tao \
        --class_list car,bicycle,person,road_sign \
        -x 1 \
        -b 8 \
        --mode DetectNet_v2 \
        -i https \
        -u localhost:8000 \
        --async \
        --output_path /path/to/the/output/directory \
        --postprocessing_config $tao_triton_root/tao_triton/python/clustering_specs/clustering_config_dashcamnet.prototxt 
  ```

3. For running an Image Classification model, the command line would be as follows:

```sh
python tao_client.py \
       /path/to/a/directory/of/images \
       -m vehicletypenet_tao \
       -x 1 \
       -b 1 \
       --mode Classification \
       -i https \
       -u localhost:8000 \
       --async \
       --output_path /path/to/the/output/directory
```

The output is generated in the `/path/to/the/output/directory/results.txt`, with in the following format.

```text
/path/to/image.jpg, 1.0000(2)= class_2, 0.0000(0)= class_0, 0.0000(3)= class_3, 0.0000(5)= class_5, 0.0000(4)= class_4, 0.0000(1)= class_1 .. 0.000(N)= class_N
```
4. For running LPRNet model, the command line would be as follows:

```sh
python tao_client.py \
       /path/to/a/directory/of/images \
       -m lprnet_tao \
       -x 1 \
       -b 1 \
       --mode LPRNet \
       -i https \
       -u localhost:8000 \
       --async \
       --output_path /path/to/the/output/directory
```
The test dataset can be downloaded from https://github.com/openalpr/benchmarks/tree/master/seg_and_ocr/usimages.
For example, run following command to download.
`wget https://github.com/openalpr/benchmarks/raw/master/seg_and_ocr/usimages/ca286.png`.
The output is generated in the `/path/to/the/output/directory/results.txt`, with in the following format.

```text
/path/to/image.jpg : ['xxxxx']
```

5. For running YOLOv3 model, the command line would be as follows:
```sh
python tao_client.py \
       /path/to/a/directory/of/images \
       -m yolov3_tao \
       -x 1 \
       -b 1 \
       --mode YOLOv3 \
       -i https \
       -u localhost:8000 \
       --async \
       --output_path /path/to/the/output/directory
```
The test image can be downloaded via following command.
`wget https://developer.nvidia.com/sites/default/files/akamai/NGC_Images/models/peoplenet/input_11ft45deg_000070.jpg`.
The infered images are generated in the `/path/to/the/output/directory/infer_images`.
The labels are generated in the `/path/to/the/output/directory/infer_labels`.

6. For running Peoplesegnet model, the command line would be as follows:
```sh
python tao_client.py \
       /path/to/a/directory/of/images \
       -m peoplesegnet_tao \
       -x 1 \
       -b 1 \
       --mode Peoplesegnet \
       -i https \
       -u localhost:8000 \
       --async \
       --output_path /path/to/the/output/directory
```
The test image can be downloaded via following command.
`wget https://developer.nvidia.com/sites/default/files/akamai/NGC_Images/models/peoplenet/input_11ft45deg_000070.jpg`.
The infered images are generated in the `/path/to/the/output/directory/infer_images`.
The labels are generated in the `/path/to/the/output/directory/infer_labels`.

7. For running Retinanet model, the command line would be as follows:
```sh
python tao_client.py \
       /path/to/a/directory/of/images \
       -m retinanet_tao \
       -x 1 \
       -b 1 \
       --mode Retinanet \
       -i https \
       -u localhost:8000 \
       --async \
       --output_path /path/to/the/output/directory
```
The test image can be downloaded via following command.
`wget https://developer.nvidia.com/sites/default/files/akamai/NGC_Images/models/peoplenet/input_11ft45deg_000070.jpg`.
The infered images are generated in the `/path/to/the/output/directory/infer_images`.
The labels are generated in the `/path/to/the/output/directory/infer_labels`.

8. For running Multitask_classification model, the command line would be as follows:
```sh
python tao_client.py \
       /path/to/a/directory/of/images \
       -m multitask_classification_tao \
       -x 1 \
       -b 1 \
       --mode Multitask_classification \
       -i https \
       -u localhost:8000 \
       --async \
       --output_path /path/to/the/output/directory
```
The test dataset can be downloaded from https://www.kaggle.com/paramaggarwal/fashion-product-images-small.
Before logining, you will need a Kaggle account. 
The inferenced results are generated in the `/path/to/the/output/directory/result.txt`.

9. For running Pose_classification model, if the input is a formatted Numpy array, the command line would be as follows:
```sh
python tao_client.py \
       /path/to/a/file/of/pose/sequences \
       -m pose_classification_tao \
       -x 1 \
       -b 1 \
       --mode Pose_classification \
       -i https \
       -u localhost:8000 \
       --async \
       --output_path /path/to/the/output/directory
```
The test dataset can be downloaded from [here](https://drive.google.com/file/d/1GhSt53-7MlFfauEZ2YkuzOaZVNIGo_c-/view?usp=sharing).
The inferenced results are generated in the `/path/to/the/output/directory/results.txt`.

The input can also be a JSON file generated from the [`deepstream-bodypose-3d`](https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps/tree/master/deepstream-bodypose-3d) app, and the command line would be as follows:
```sh
python tao_client.py \
       /path/to/a/json/file/of/pose/metadata \
       --dataset_convert_config /path/to/a/yaml/file/of/dataset/convert/config \
       -m pose_classification_tao \
       -x 1 \
       -b 1 \
       --mode Pose_classification \
       -i https \
       -u localhost:8000 \
       --async \
       --output_path /path/to/the/output/directory
```
The inferenced results are generated in the `/path/to/the/output/directory/results.json`, which follow the same format as the JSON metadata from `deepstream-bodypose-3d` and add predicted `"action"` to each object at each frame. 

Note that the frame sequence of each object is broken into skeleton segments by a dataset converter (see the figure below). The `sequence_length` and `sequence_overlap` are configurable in [`dataset_convert_config_pose_classification.yaml`](/tao_triton/python/dataset_convert_specs/dataset_convert_config_pose_classification.yaml). The output labels are assigned to frames after certain period of time.

<div align="center">
  <img src = "/images/pose_cls_e2e_inference.png", width="960">
</div>

To perform end-to-end inference, run the following quick start script to start the server (only the Pose Classification model will be downloaded and converted):

 ```sh
 bash scripts/pose_cls_e2e_inference/start_server.sh
 ```

 Then run the following script for body pose estimation from DeepStream and sending an inference request to the server:

 ```sh
 bash scripts/pose_cls_e2e_inference/start_client.sh
 ```

10. For running Re_identification model, the command line would be as follows:
```sh
python tao_client.py \
       /path/to/a/directory/of/query/images \
       --test_dir /path/to/a/directory/of/test/images \
       -m re_identification_tao \
       -x 1 \
       -b 16 \
       --mode Re_identification \
       -i https \
       -u localhost:8000 \
       --async \
       --output_path /path/to/the/output/directory
```
The test dataset can be downloaded from [here](https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html).
The inferenced results are generated in the `/path/to/the/output/directory/results.json`.

To perform end-to-end inference, run the following quick start script to start the server (only the Re-Identification model will be downloaded and converted):

 ```sh
 bash scripts/re_id_e2e_inference/start_server.sh
 ```

 Then run the following script for re-identification and sending an inference request to the server:

 ```sh
 bash scripts/re_id_e2e_inference/start_client.sh
 ```

 11. For running VisualChangeNet model, the command line would be as follows:

```sh
python tao_triton/python/entrypoints/tao_client.py \
       /path/to/a/directory/of/images \
       -m visual_changenet_segmentation_tao \
       -x 1 \
       -b 1 \
       --mode VisualChangeNet \
       -i https \
       -u localhost:8000 \
       --async \
       --output_path /path/to/the/output/directory \
       --img_dirs "A" \
       --img_dirs "B"
```

The infered images are generated in the `/path/to/the/output/directory/`

 12. For running CenterPose model, the command line would be as follows:

- Note that please make sure the camera calibration infomation is correct before running the command line in `clustering_config_centerpose.prototxt`. 

```sh
python tao_triton/python/entrypoints/tao_client.py \
       /path/to/a/directory/of/images \
       -m centerpose_tao \
       -x 1 \
       -b 1 \
       --mode CenterPose \
       -i https \
       -u localhost:8000 \
       --async \
       --output_path /path/to/the/output/directory \
       --postprocessing_config $tao_triton_root/tao_triton/python/clustering_specs/clustering_config_centerpose.prototxt
```

The infered images and the related json files are generated in the `/path/to/the/output/directory/`