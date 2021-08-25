# Configuring the Client Samples

- [DetectNet_v2](#detectnet-v2)
  - [Configuring the DetectNet_v2 model entry in the model repository](#configuring-the-detectnet-v2-model-entry-in-the-model-repository)
  - [Configuring the Post-processor](#configuring-the-post-processor)
- [Classification](#classification)
  - [Configuring the Classification model entry in the model repository](#configuring-the-classification-model-entry-in-the-model-repository)

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
