# Configuring the Client Samples

- [DetectNet_v2](#detectnet-v2)
  - [Configuring the DetectNet_v2 model entry in the model repository](#configuring-the-detectnet-v2-model-entry-in-the-model-repository)
  - [Configuring the Post-processor](#configuring-the-post-processor)
- [Classification](#classification)
  - [Configuring the Classification model entry in the model repository](#configuring-the-classification-model-entry-in-the-model-repository)
- [LPRNet](#lprnet)
  - [Configuring the LPRNet model entry in the model repository](#configuring-the-lprnet-model-entry-in-the-model-repository)
  - [Configuring the LPRNet model Post-processor](#configuring-the-lprnet-model-post-processor)
- [YOLOv3](#yolov3)
  - [Configuring the YOLOv3 model entry in the model repository](#configuring-the-yolov3-model-entry-in-the-model-repository)
  - [Configuring the YOLOv3 model Post-processor](#configuring-the-yolov3-model-post-processor)
- [Peoplesegnet](#peoplesegnet)
  - [Configuring the Peoplesegnet model entry in the model repository](#configuring-the-peoplesegnet-model-entry-in-the-model-repository)
  - [Configuring the Peoplesegnet model Post-processor](#configuring-the-peoplesegnet-model-post-processor)
- [Retinanet](#retinanet)
  - [Configuring the Retinanet model entry in the model repository](#configuring-the-retinanet-model-entry-in-the-model-repository)
  - [Configuring the Retinanet model Post-processor](#configuring-the-retinanet-model-post-processor)
- [Multitask_classification](#multitask_classification)
  - [Configuring the Multitask_classification model entry in the model repository](#configuring-the-multitask_classification-model-entry-in-the-model-repository)
  - [Configuring the Multitask_classification model Post-processor](#configuring-the-multitask_classification-model-post-processor)
- [Pose_classification](#pose_classification)
  - [Configuring the Pose_classification model entry in the model repository](#configuring-the-pose_classification-model-entry-in-the-model-repository)
  - [Configuring the Pose_classification model Post-processor](#configuring-the-pose_classification-model-post-processor)
  - [Configuring the Pose_classification data converter](#configuring-the-pose_classification-data-converter)

The inference client samples provided in this provide several parameters that the user can configure.
This section elaborates about those parameters in more detail.

## DetectNet_v2

The DetectNet_v2 inference sample has 2 components that can be configured

1. [Model Repository](#configuring-the-detectnet-v2-model-entry-in-the-model-repository)
2. [Post Processor](#configuring-the-post-processor)

### Configuring the DetectNet_v2 model entry in the model repository

The model repository is the location on the Triton Server, where the model served from. Triton expects the models
in the model repository to be follow the layout defined [here](https://github.com/triton-inference-server/server/blob/main/docs/model_repository.md#repository-layout).

A sample model repository for a DetectNet_v2 PeopleNet model would have the following contents.

```text
model_repository_root/
    peoplenet_tao/
        config.pbtxt
        1/
            model.plan
```

The `config.pbtxt` file, describes the model configuration for the model. A sample model configuration file for the PeopleNet
model would look like this.

```proto
name: "peoplenet_tao"
platform: "tensorrt_plan"
max_batch_size: 16
input [
    {
        name: "input_1"
        data_type: TYPE_FP32
        format: FORMAT_NCHW
        dims: [ 3, 544, 960 ]
    }
]
output [
    {
        name: "output_bbox/BiasAdd"
        data_type: TYPE_FP32
        dims: [ 12, 34, 60 ]
    },
    {
        name: "output_cov/Sigmoid"
        data_type: TYPE_FP32
        dims: [ 3, 34, 60 ]
    }
]
dynamic_batching { }
```

The following table explains the parameters in the config.pbtxt

| **Parameter Name** | **Description** | **Type**  | **Supported Values**| **Sample Values**|
| :----              | :-------------- | :-------: | :------------------ | :--------------- |
| name | The user readable name of the served model | string |   | peoplenet_tao|
| platform | The backend used to parse and run the model | string | tensorrt_plan | tensorrt_plan |
| max_batch_size | The maximum batch size used to create the TensorRT engine.<br>This should be the same as the `max_batch_size` parameter of the `tao-converter`| int |  | 16 |
| input | Configuration elements for the input nodes | list of protos/node |  |  |
| output | Configuration elements for the output nodes | list of protos/node |  |  |
| dynamic_batching | Configuration element to enable [dynamic batching](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#dynamic-batcher) using Triton | proto element |  |  |

The input and output elements in the config.pbtxt provide the configurable parameters for the input and output nodes of the model
that is being served. As seen in the sample, a detectnet_v2 model has 1 input node ("input_1") and 2 outputs nodes, namely:

- `output_bbox/BiasAdd`
- `output_cov/Sigmoid`

All the parameters defined the `input` and `output` elements remain the same for any DetectNet_v2 model trained
using TAO Toolkit, except for dims. You may derive the dimensions of the input and output nodes as follows:

1. For `input_1`, the parameter `dims` is the input dimensions of the model in C, H, W order (where C = Channels, H = height, W = Width).
   This parameter should be indentical to the dimension mentioned to `-d` option of the `tao-converter`.

2. For `output_cov/Sigmoid`, the parameter `dims` is the output dimension of the coverage blob in C, H, W order. The value for the dimensions can
   be calculated as C = num_classes, H = ceil(input_h/model_stride) , W = ceil(input_w/model_stride)

3. For `output_bbox/BiasAdd`, the parameter `dims` is the output dimension of the coverage blob in C, H, W order. The value for the dimensions can
   be calculated as C = num_classes * 4, H = ceil(input_h/model_stride) , W = ceil(input_w/model_stride)

   >
   > Note:
   >
   >`model_stride=16` for all combinations of backbones with DetectNet_v2, except for a
   > Detectnet_v2 model with `efficientnet_b0` where the `model_stride=32`. For a complete
   > list of backbone supported refer to the [TAO Toolkit documentation](https://docs.nvidia.com/tao/tao-toolkit/text/open_images/overview.html).

### Configuring the Post-processor

The DetectNet_v2 model generates raw output tensors which needs to be post-processed to be able generate renderable
bounding boxes. The reference implementation of the post-processor is defined in the [here](../python/postprocessing/detectnet_processor.py).

A sample configuration file to configure the postprocessor module of a PeopleNet `DetectNet_v2` look as shown below

```proto
linewidth: 4
stride: 16
classwise_clustering_config{
    key: "person"
    value: {
        coverage_threshold: 0.005
        minimum_bounding_box_height: 4
        dbscan_config{
            dbscan_eps: 0.3
            dbscan_min_samples: 0.05
            dbscan_confidence_threshold: 0.9
        }
        bbox_color{
            R: 0
            G: 255
            B: 0
        }
    }
}
classwise_clustering_config{
    key: "bag"
    value: {
        coverage_threshold: 0.005
        minimum_bounding_box_height: 4
        dbscan_config{
            dbscan_eps: 0.3
            dbscan_min_samples: 0.05
            dbscan_confidence_threshold: 0.9
        }
        bbox_color{
            R: 0
            G: 255
            B: 255
        }
    }
}
classwise_clustering_config{
    key: "face"
    value: {
        coverage_threshold: 0.005
        minimum_bounding_box_height: 4
        dbscan_config{
            dbscan_eps: 0.3
            dbscan_min_samples: 0.05
            dbscan_confidence_threshold: 0.2
        }
        bbox_color{
            R: 255
            G: 0
            B: 0
        }
    }
}
```

The following table explains the configurable elements of the postprocessor plugin.

| **Parameter Name** | **Description** | **Type**  | **Supported Values**| **Sample Values**|
| :----              | :-------------- | :-------: | :------------------ | :--------------- |
| linewidth | The width of the bounding box edges | int | >0 | 2 |
| stride | The ratio of the input shape to output shape of the model. <br>This value is 32 only for the `efficientNet_b0` backbone with DetectNet_v2 | int | 16,32 | 16 |
| classwise_clustering_config | Dictionary proto element, defining clustering parameters per class| dict | -  | - |

For each object class that the DetectNet_v2 network generates an output tensor, there is a `classwise_clustering_config`,
element that defines the clustering parameters for this class.

| **Parameter**                | **Datatype** | **Default** | **Description** | **Supported Values**     |
|------------------------------|--------------|-------------|-----------------|--------------------------|
| coverage_threshold           | float        |             | The minimum threshold of the coverage tensor output to be considered a valid candidate box for clustering.<br>The four coordinates from the bbox tensor at the corresponding indices are passed for clustering | 0.0 - 1.0 |
| minimum_bounding_box_height  | int          | --          | The minimum height in pixels to consider as a valid detection post clustering.                              | 0 - input image height   |
| bbox_color  | BboxColor Proto Object        | None        | RGB channel wise color intensity per box.                  | R: 0 - 255 <br> G: 0 - 255 <br> B: 0 - 255     |
| dbscan_config | DBSCANConfig Proto Object   | None        | Proto object to configure the DBSCAN post processor plugin for the networks | - |

The table below expands the configurable parameters defined under the `dbscan_config` element.

| **Parameter**                | **Datatype** | **Default** | **Description** | **Supported Values**     |
|------------------------------|--------------|-------------|-----------------|--------------------------|
| dbscan_eps                   | float        |             | The maximum distance between two samples for one to be considered in the neighborhood of the other. <br>This is not a maximum bound on the distances of points within a cluster. The greater the `dbscan_eps` value, the more boxes are grouped together. | 0.0 - 1.0 |
| dbscan_min_samples           | float        | --          | The total weight in a neighborhood for a point to be considered as a core point. <br>This includes the point itself. | 0.0 - 1.0  |
| dbscan_confidence_threshold  | float        | 0.1         | The confidence threshold used to filter out the clustered bounding box output from DBSCAN.                  | > 0.0                    |

> Note:
>
> A unique key-value entry has to be defined for every class that the DetectNet_v2 model is trained for. <br>Please refer to the [DetectNet_v2 documentation](https://docs.nvidia.com/tao/tao-toolkit/text/object_detection/detectnet_v2.html#exporting-the-detectnet-v2-model)
> for more information on how to derive the class labels from the training configuration file of the network at export.

The post processor configuration in a protobuf file, who's schema is defined in this [file](../python/proto/postprocessor_config.proto).

## Classification

The Classification inference sample has 1 component that can be configured

1. [Model Repository](#classification-model-repository)

### Configuring the Classification model entry in the model repository

The model repository is the location on the Triton Server, where the model served from. Triton expects the models
in the model repository to be follow the layout defined [here](https://github.com/triton-inference-server/server/blob/main/docs/model_repository.md#repository-layout).

A sample model repository for an image classification VehicleTypeNet model would have the following contents.

```text
model_repository_root/
    vehicletypenet_tao/
        config.pbtxt
        labels.txt
        1/
            model.plan
```

The `config.pbtxt` file, describes the model configuration for the model. A sample model configuration file for the VehicleTypeNet
model would look like this.

```proto
name: "vehicletypenet_tao"
platform: "tensorrt_plan"
max_batch_size : 1
input [
    {
        name: "input_1"
        data_type: TYPE_FP32
        format: FORMAT_NCHW
        dims: [ 3, 224, 224 ]
    }
]
output [
    {
        name: "predictions/Softmax"
        data_type: TYPE_FP32
        dims: [6, 1, 1]
        label_filename: "labels.txt"
    }
]
dynamic_batching { }
```

The following table explains the parameters in the config.pbtxt

| **Parameter Name** | **Description** | **Type**  | **Supported Values**| **Sample Values**|
| :----              | :-------------- | :-------: | :------------------ | :--------------- |
| name | The user readable name of the served model | string |   | peoplenet_tao|
| platform | The backend used to parse and run the model | string | tensorrt_plan | tensorrt_plan |
| max_batch_size | The maximum batch size used to create the TensorRT engine.<br>This should be the same as the `max_batch_size` parameter of the `tao-converter`| int |  | 16 |
| input | Configuration elements for the input nodes | list of protos/node |  |  |
| output | Configuration elements for the output nodes | list of protos/node |  |  |
| dynamic_batching | Configuration element to enable [dynamic batching](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#dynamic-batcher) using Triton | proto element |  |  |

The input and output elements in the config.pbtxt provide the configurable parameters for the input and output nodes of the model
that is being served. As seen in the sample, a classification model has 1 input node `input_1` and 1 output node `predictions/Softmax`

All the parameters defined the `input` and `output` elements remain the same for any image classification model trained
using TAO Toolkit, except for the dims. You may derive the dimensions of the input and output nodes as follows:

1. For `input_1`, the parameter `dims` is the input dimensions of the model in C, H, W order (where C = Channels, H = height, W = Width).
   This parameter should be indentical to the dimension mentioned to `-d` option of the `tao-converter`.

2. For `predictions/Softmax`, the parameter `dims` is the output dimension of the coverage blob in C, H, W order. The value for the dimensions can
   be calculated as C = number of classes, H = 1 , W = 1

## LPRNet

The LPRNet inference sample has 2 component that can be configured

1. [Model Repository](#lprnet-model-repository)
2. [Configuring the LPRNet model Post-processor](#configuring-the-lprnet-model-post-processor)

### Configuring the LPRNet model entry in the model repository

The model repository is the location on the Triton Server, where the model served from. Triton expects the models
in the model repository to be follow the layout defined [here](https://github.com/triton-inference-server/server/blob/main/docs/model_repository.md#repository-layout).

A sample model repository for an LPRnet model would have the following contents.

```text
model_repository_root/
    lprnet_tao/
        config.pbtxt
        dict_us.txt
        1/
            model.plan
```

The `config.pbtxt` file, describes the model configuration for the model. A sample model configuration file for the LPRNet
model would look like this.

```proto
name: "lprnet_tao"
platform: "tensorrt_plan"
max_batch_size : 16
input [
    {
        name: "image_input"
        data_type: TYPE_FP32
        format: FORMAT_NCHW
        dims: [ 3, 48, 96 ]
    }
]
output [
    {
        name: "tf_op_layer_ArgMax"
        data_type: TYPE_INT32
        dims: [ 24 ]
    },
    {
        name: "tf_op_layer_Max"
        data_type: TYPE_FP32
        dims: [ 24 ]
    }

]
dynamic_batching { }
```

The following table explains the parameters in the config.pbtxt

| **Parameter Name** | **Description** | **Type**  | **Supported Values**| **Sample Values**|
| :----              | :-------------- | :-------: | :------------------ | :--------------- |
| name | The user readable name of the served model | string |   | lprnet_tao|
| platform | The backend used to parse and run the model | string | tensorrt_plan | tensorrt_plan |
| max_batch_size | The maximum batch size used to create the TensorRT engine.<br>This should be the same as the `max_batch_size` parameter of the `tao-converter`| int |  | 16 |
| input | Configuration elements for the input nodes | list of protos/node |  |  |
| output | Configuration elements for the output nodes | list of protos/node |  |  |
| dynamic_batching | Configuration element to enable [dynamic batching](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#dynamic-batcher) using Triton | proto element |  |  |

The input and output elements in the config.pbtxt provide the configurable parameters for the input and output nodes of the model
that is being served. As seen in the sample, a lprnet model has 1 input node `image_input` and 2 output node `tf_op_layer_ArgMax`
and `tf_op_layer_Max`.
The dims in output node should the 1/4 of the width in the dims of input node. For example, as above, 24 equals to 1/4 of 96.

### Configuring the LPRnet model Post-processor

Please generate characters list file under `model_repository/lprnet_tao` folder. The file name should be characters_list.txt.
A sample file for US license plate would look like this

```proto
0
1
2
3
4
5
6
7
8
9
A
B
C
D
E
F
G
H
I
J
K
L
M
N
P
Q
R
S
T
U
V
W
X
Y
Z
```
This characters_list.txt file contains all the characters found in license plate dataset. Each character occupies one line.

## YOLOv3

The YOLOv3 inference sample has 2 component that can be configured

1. [Model Repository](#yolov3-model-repository)
2. [Configuring the YOLOv3 model Post-processor](#configuring-the-yolov3-model-post-processor)

### Configuring the YOLOv3 model entry in the model repository

The model repository is the location on the Triton Server, where the model served from. Triton expects the models
in the model repository to be follow the layout defined [here](https://github.com/triton-inference-server/server/blob/main/docs/model_repository.md#repository-layout).

A sample model repository for an YOLOv3 model would have the following contents.

```text
model_repository_root/
    yolov3_tao/
        config.pbtxt
        1/
            model.plan
```

The `config.pbtxt` file, describes the model configuration for the model. A sample model configuration file for the YOLOv3
model would look like this.

```proto
name: "yolov3_tao"
platform: "tensorrt_plan"
max_batch_size: 16
input [
  {
    name: "Input"
    data_type: TYPE_FP32
    format: FORMAT_NCHW
    dims: [ 3, 544, 960 ]
  }
]
output [
  {
    name: "BatchedNMS"
    data_type: TYPE_INT32
    dims: [ 1 ]
  },
  {
    name: "BatchedNMS_1"
    data_type: TYPE_FP32
    dims: [ 200, 4 ]
  },
  {
    name: "BatchedNMS_2"
    data_type: TYPE_FP32
    dims: [ 200 ]
  },
  {
    name: "BatchedNMS_3"
    data_type: TYPE_FP32
    dims: [ 200 ]
  }
]
dynamic_batching { }
```

The following table explains the parameters in the config.pbtxt

| **Parameter Name** | **Description** | **Type**  | **Supported Values**| **Sample Values**|
| :----              | :-------------- | :-------: | :------------------ | :--------------- |
| name | The user readable name of the served model | string |   | yolov3_tao|
| platform | The backend used to parse and run the model | string | tensorrt_plan | tensorrt_plan |
| max_batch_size | The maximum batch size used to create the TensorRT engine.<br>This should be the same as the `max_batch_size` parameter of the `tao-converter`| int |  | 16 |
| input | Configuration elements for the input nodes | list of protos/node |  |  |
| output | Configuration elements for the output nodes | list of protos/node |  |  |
| dynamic_batching | Configuration element to enable [dynamic batching](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#dynamic-batcher) using Triton | proto element |  |  |

The input and output elements in the config.pbtxt provide the configurable parameters for the input and output nodes of the model
that is being served. As seen in the sample, a yolov3 model has 1 input node `Input` and 4 output node `BatchedNMS` , `BatchedNMS_1` , `BatchedNMS_2`
and `BatchedNMS_3`.

### Configuring the YOLOv3 model Post-processor

Refer to `model_repository/yolov3_tao` folder. 

## Peoplesegnet

The Peoplesegnet inference sample has 2 component that can be configured

1. [Model Repository](#peoplesegnet-model-repository)
2. [Configuring the Peoplesegnet model Post-processor](#configuring-the-peoplesegnet-model-post-processor)

### Configuring the Peoplesegnet model entry in the model repository

The model repository is the location on the Triton Server, where the model served from. Triton expects the models
in the model repository to be follow the layout defined [here](https://github.com/triton-inference-server/server/blob/main/docs/model_repository.md#repository-layout).

A sample model repository for an Peoplesegnet model would have the following contents.

```text
model_repository_root/
    peoplesegnet_tao/
        config.pbtxt
        1/
            model.plan
```

The `config.pbtxt` file, describes the model configuration for the model. A sample model configuration file for the Peoplesegnet
model would look like this.

```proto
name: "peoplesegnet_tao"
platform: "tensorrt_plan"
max_batch_size: 16
input [
  {
    name: "Input"
    data_type: TYPE_FP32
    format: FORMAT_NCHW
    dims: [ 3, 576, 960 ]
  }
]
output [
  {
    name: "generate_detections"
    data_type: TYPE_FP32
    dims: [ 100, 6 ]
  },
  {
    name: "mask_fcn_logits/BiasAdd"
    data_type: TYPE_FP32
    dims: [ 100, 2, 28, 28 ]
  }
]
dynamic_batching { }
```

The following table explains the parameters in the config.pbtxt

| **Parameter Name** | **Description** | **Type**  | **Supported Values**| **Sample Values**|
| :----              | :-------------- | :-------: | :------------------ | :--------------- |
| name | The user readable name of the served model | string |   | peoplesegnet_tao|
| platform | The backend used to parse and run the model | string | tensorrt_plan | tensorrt_plan |
| max_batch_size | The maximum batch size used to create the TensorRT engine.<br>This should be the same as the `max_batch_size` parameter of the `tao-converter`| int |  | 16 |
| input | Configuration elements for the input nodes | list of protos/node |  |  |
| output | Configuration elements for the output nodes | list of protos/node |  |  |
| dynamic_batching | Configuration element to enable [dynamic batching](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#dynamic-batcher) using Triton | proto element |  |  |

The input and output elements in the config.pbtxt provide the configurable parameters for the input and output nodes of the model
that is being served. As seen in the sample, a peoplesegnet model has 1 input node `Input` and 2 output node `generate_detections` and `mask_fcn_logits/BiasAdd`.

### Configuring the Peoplesegnet model Post-processor

Refer to `model_repository/peoplesegnet_tao` folder. 

## Retinanet

The Retinanet inference sample has 2 component that can be configured

1. [Model Repository](#retinanet-model-repository)
2. [Configuring the Retinanet model Post-processor](#configuring-the-retinanet-model-post-processor)

### Configuring the Retinanet model entry in the model repository

The model repository is the location on the Triton Server, where the model served from. Triton expects the models
in the model repository to be follow the layout defined [here](https://github.com/triton-inference-server/server/blob/main/docs/model_repository.md#repository-layout).

A sample model repository for an Retinanet model would have the following contents.

```text
model_repository_root/
    retinanet_tao/
        config.pbtxt
        1/
            model.plan
```

The `config.pbtxt` file, describes the model configuration for the model. A sample model configuration file for the Retinanet
model would look like this.

```proto
name: "retinanet_tao"
platform: "tensorrt_plan"
max_batch_size: 16
input [
  {
    name: "Input"
    data_type: TYPE_FP32
    format: FORMAT_NCHW
    dims: [ 3, 544, 960 ]
  }
]
output [
  {
    name: "NMS"
    data_type: TYPE_FP32
    dims: [ 1, 250, 7 ]
  },
  {
    name: "NMS_1"
    data_type: TYPE_FP32
    dims: [ 1, 1, 1 ]
  }
]
dynamic_batching { }
```

The following table explains the parameters in the config.pbtxt

| **Parameter Name** | **Description** | **Type**  | **Supported Values**| **Sample Values**|
| :----              | :-------------- | :-------: | :------------------ | :--------------- |
| name | The user readable name of the served model | string |   | retinanet_tao|
| platform | The backend used to parse and run the model | string | tensorrt_plan | tensorrt_plan |
| max_batch_size | The maximum batch size used to create the TensorRT engine.<br>This should be the same as the `max_batch_size` parameter of the `tao-converter`| int |  | 16 |
| input | Configuration elements for the input nodes | list of protos/node |  |  |
| output | Configuration elements for the output nodes | list of protos/node |  |  |
| dynamic_batching | Configuration element to enable [dynamic batching](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#dynamic-batcher) using Triton | proto element |  |  |

The input and output elements in the config.pbtxt provide the configurable parameters for the input and output nodes of the model
that is being served. As seen in the sample, a retinanet model has 1 input node `Input` and 2 output node `NMS` and `NMS_1`.

### Configuring the Retinanet model Post-processor

Refer to `model_repository/retinanet_tao` folder. 

## Multitask_classification

The Multitask_classification inference sample has 2 component that can be configured

1. [Model Repository](#multitask_classification-model-repository)
2. [Configuring the Multitask_classification model Post-processor](#configuring-the-multitask_classification-model-post-processor)

### Configuring the Multitask_classification model entry in the model repository

The model repository is the location on the Triton Server, where the model served from. Triton expects the models
in the model repository to be follow the layout defined [here](https://github.com/triton-inference-server/server/blob/main/docs/model_repository.md#repository-layout).

A sample model repository for an Multitask_classification model would have the following contents.

```text
model_repository_root/
    multitask_classification_tao/
        config.pbtxt
        1/
            model.plan
```

The `config.pbtxt` file, describes the model configuration for the model. A sample model configuration file for the Multitask_classification
model would look like this.

```proto
name: "multitask_classification_tao"
platform: "tensorrt_plan"
max_batch_size: 16
input [
  {
    name: "input_1"
    data_type: TYPE_FP32
    format: FORMAT_NCHW
    dims: [ 3, 80, 60 ]
  }
]
output [
  {
    name: "season/Softmax"
    data_type: TYPE_FP32
    dims: [ 4, 1, 1 ]
  },
  {
    name: "category/Softmax"
    data_type: TYPE_FP32
    dims: [ 10, 1, 1 ]
  },
  {
    name: "base_color/Softmax"
    data_type: TYPE_FP32
    dims: [ 11, 1, 1 ]
  }
]
dynamic_batching { }
```

The following table explains the parameters in the config.pbtxt

| **Parameter Name** | **Description** | **Type**  | **Supported Values**| **Sample Values**|
| :----              | :-------------- | :-------: | :------------------ | :--------------- |
| name | The user readable name of the served model | string |   | multitask_classification_tao|
| platform | The backend used to parse and run the model | string | tensorrt_plan | tensorrt_plan |
| max_batch_size | The maximum batch size used to create the TensorRT engine.<br>This should be the same as the `max_batch_size` parameter of the `tao-converter`| int |  | 16 |
| input | Configuration elements for the input nodes | list of protos/node |  |  |
| output | Configuration elements for the output nodes | list of protos/node |  |  |
| dynamic_batching | Configuration element to enable [dynamic batching](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#dynamic-batcher) using Triton | proto element |  |  |

The input and output elements in the config.pbtxt provide the configurable parameters for the input and output nodes of the model
that is being served. As seen in the sample, a Multitask_classification model has 1 input node `input_1` and 3 output node `season/Softmax` , `category/Softmax` and `base_color/Softmax`.

### Configuring the Multitask_classification model Post-processor

Refer to `model_repository/multitask_classification_tao` folder. 

## Pose_classification

The Pose_classification inference sample has 3 components that can be configured

1. [Model Repository](#configuring-the-pose_classification-model-entry-in-the-model-repository)
2. [Configuring the Pose_classification model Post-processor](#configuring-the-pose_classification-model-post-processor)
3. [Configuring the Pose_classification data converter](#configuring-the-pose_classification-data-converter)

### Configuring the Pose_classification model entry in the model repository

The model repository is the location on the Triton Server, where the model served from. Triton expects the models
in the model repository to be follow the layout defined [here](https://github.com/triton-inference-server/server/blob/main/docs/model_repository.md#repository-layout).

A sample model repository for an Pose_classification model would have the following contents.

```text
model_repository_root/
    pose_classification_tao/
        config.pbtxt
        labels.txt
        1/
            model.plan
```

The `config.pbtxt` file, describes the model configuration for the model. A sample model configuration file for the Pose_classification
model would look like this.

```proto
name: "pose_classification_tao"
platform: "tensorrt_plan"
max_batch_size: 16
input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ 3, 300, 34, 1 ]
  }
]
output [
  {
    name: "fc_pred"
    data_type: TYPE_FP32
    dims: [ 6 ]
    label_filename: "labels.txt"
  }
]
dynamic_batching { }
```

The following table explains the parameters in the config.pbtxt

| **Parameter Name** | **Description** | **Type**  | **Supported Values**| **Sample Values**|
| :----              | :-------------- | :-------: | :------------------ | :--------------- |
| name | The user readable name of the served model | string |   | pose_classification_tao|
| platform | The backend used to parse and run the model | string | tensorrt_plan | tensorrt_plan |
| max_batch_size | The maximum batch size used to create the TensorRT engine.<br>This should be the same as the `max_batch_size` parameter of the `tao-converter`| int |  | 16 |
| input | Configuration elements for the input nodes | list of protos/node |  |  |
| output | Configuration elements for the output nodes | list of protos/node |  |  |
| dynamic_batching | Configuration element to enable [dynamic batching](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#dynamic-batcher) using Triton | proto element |  |  |

The input and output elements in the config.pbtxt provide the configurable parameters for the input and output nodes of the model
that is being served. As seen in the sample, a Pose_classification model has 1 input node `input` and 1 output node `fc_pred`.

### Configuring the Pose_classification model Post-processor

Refer to `model_repository/pose_classification_tao` folder. 

### Configuring the Pose_classification data converter

When the input is a JSON file generated from the [deepstream-bodypose-3d](https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps/tree/master/deepstream-bodypose-3d) app, it needs to be converted into skeleton sequences to be consumed by the Pose_classification model.

A sample [configuration file](/tao_triton/python/dataset_convert_specs/dataset_convert_config_pose_classification.yaml) to configure the dataset converter of Pose Classification looks as shown below

```yaml
pose_type: "3dbp"
num_joints: 34
frame_width: 1920
frame_height: 1080
focal_length: 1200.0
sequence_length_max: 300
sequence_length_min: 10
sequence_length: 100
sequence_overlap: 0.5
```

The following table explains the configurable parameters of the dataset converter.

| **Parameter Name** | **Description** | **Type**  | **Supported Values**| **Sample Values**|
| :----              | :-------------- | :-------: | :------------------ | :--------------- |
| pose_type | The type of body pose | string | 3dbp, 25dbp, or 2dbp | 3dbp|
| num_joints | The total number of joints in the skeleton graph layout | int |  | 34 |
| frame_width | The width of the video frame in pixel | int |  | 1920 |
| frame_height | The height of the video frame in pixel | int |  | 1080 |
| focal_length | The focal length that the video was captured in | float |  | 1200.0 |
| sequence_length_max | The maximum sequence length in frame | int |  | 300 |
| sequence_length_min | The minimum sequence length in frame | int |  | 10 |
| sequence_length | The sequence length for sampling sequences | int |  | 100 |
| sequence_overlap | The overlap between sequences during samping | float |  | 0.5 |
