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

import argparse
from copy import deepcopy
from functools import partial
import logging
import os
import sys

from attrdict import AttrDict
import numpy as np
from PIL import Image
from tqdm import tqdm

import tritonclient.grpc as grpcclient
import tritonclient.grpc.model_config_pb2 as mc
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from tritonclient.utils import triton_to_np_dtype

from tao_triton.python.types import Frame, UserData
from tao_triton.python.postprocessing.detectnet_processor import DetectNetPostprocessor
from tao_triton.python.postprocessing.classification_postprocessor import ClassificationPostprocessor
from tao_triton.python.postprocessing.lprnet_postprocessor import LPRPostprocessor
from tao_triton.python.postprocessing.yolov3_postprocessor import YOLOv3Postprocessor
from tao_triton.python.postprocessing.peoplesegnet_postprocessor import PeoplesegnetPostprocessor
from tao_triton.python.postprocessing.retinanet_postprocessor import RetinanetPostprocessor
from tao_triton.python.postprocessing.multitask_classification_postprocessor import MultitaskClassificationPostprocessor
from tao_triton.python.utils.kitti import write_kitti_annotation
from tao_triton.python.model.detectnet_model import DetectnetModel
from tao_triton.python.model.classification_model import ClassificationModel
from tao_triton.python.model.lprnet_model import LPRModel
from tao_triton.python.model.yolov3_model import YOLOv3Model
from tao_triton.python.model.peoplesegnet_model import PeoplesegnetModel
from tao_triton.python.model.retinanet_model import RetinanetModel
from tao_triton.python.model.multitask_classification_model import MultitaskClassificationModel
from tao_triton.python.utils.raw_metrics import RawMetrics

logger = logging.getLogger(__name__)

TRITON_MODEL_DICT = {
    "classification": ClassificationModel,
    "detectnet_v2": DetectnetModel,
    "lprnet": LPRModel,
    "yolov3": YOLOv3Model,
    "peoplesegnet": PeoplesegnetModel,
    "retinanet": RetinanetModel,
    "multitask_classification":MultitaskClassificationModel
}

POSTPROCESSOR_DICT = {
    "classification": ClassificationPostprocessor,
    "detectnet_v2": DetectNetPostprocessor,
    "lprnet": LPRPostprocessor,
    "yolov3": YOLOv3Postprocessor,
    "peoplesegnet": PeoplesegnetPostprocessor,
    "retinanet": RetinanetPostprocessor,
    "multitask_classification": MultitaskClassificationPostprocessor
}


def completion_callback(user_data, result, error):
    """Callback function used for async_stream_infer()."""
    user_data._completed_requests.put((result, error))


def convert_http_metadata_config(_metadata, _config):
    """Convert to the http metadata to class Dict."""
    _model_metadata = AttrDict(_metadata)
    _model_config = AttrDict(_config)

    return _model_metadata, _model_config


def requestGenerator(batched_image_data, input_name, output_name, dtype, protocol,
                     num_classes=0):
    """Generator for triton inference requests.

    Args:
        batch_image_data (np.ndarray): Numpy array of a batch of images.
        input_name (str): Name of the input array
        output_name (list(str)): Name of the model outputs
        dtype: Tensor data type for Triton
        protocol (str): The protocol used to communicated between the Triton
            server and TAO Toolkit client.
        num_classes (int): The number of classes in the network.

    Yields:
        inputs
        outputs
        made_name (str): Name of the triton model
        model_version (int): Version number
    """
    if protocol == "grpc":
        client = grpcclient
    else:
        client = httpclient

    # Set the input data
    inputs = [client.InferInput(input_name, batched_image_data.shape, dtype)]
    inputs[0].set_data_from_numpy(batched_image_data)

    outputs = [
        client.InferRequestedOutput(
            out_name, class_count=num_classes
        ) for out_name in output_name
    ]

    yield inputs, outputs


def parse_command_line(args=None):
    """Parsing command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-a',
                        '--async',
                        dest="async_set",
                        action="store_true",
                        required=False,
                        default=False,
                        help='Use asynchronous inference API')
    parser.add_argument('--streaming',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Use streaming inference API. ' +
                        'The flag is only available with gRPC protocol.')
    parser.add_argument('-m',
                        '--model-name',
                        type=str,
                        required=True,
                        help='Name of model')
    parser.add_argument('-x',
                        '--model-version',
                        type=str,
                        required=False,
                        default="",
                        help='Version of model. Default is to use latest version.')
    parser.add_argument('-b',
                        '--batch-size',
                        type=int,
                        required=False,
                        default=1,
                        help='Batch size. Default is 1.')
    parser.add_argument('--mode',
                        type=str,
                        choices=['Classification', "DetectNet_v2", "LPRNet", "YOLOv3", "Peoplesegnet", "Retinanet", "Multitask_classification"],
                        required=False,
                        default='NONE',
                        help='Type of scaling to apply to image pixels. Default is NONE.')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8000',
                        help='Inference server URL. Default is localhost:8000.')
    parser.add_argument('-i',
                        '--protocol',
                        type=str,
                        required=False,
                        default='HTTP',
                        help='Protocol (HTTP/gRPC) used to communicate with ' +
                        'the inference service. Default is HTTP.')
    parser.add_argument('image_filename',
                        type=str,
                        nargs='?',
                        default=None,
                        help='Input image / Input folder.')
    parser.add_argument('--class_list',
                        type=str,
                        default="car,bus,truck,motorcycle,pedestrian,bicycle",
                        help="Comma separated class names",
                        required=False)
    parser.add_argument('--output_path',
                        type=str,
                        default=os.path.join(os.getcwd(), "outputs"),
                        help="Path to where the inferenced outputs are stored.",
                        required=True)
    parser.add_argument("--postprocessing_config",
                        type=str,
                        default="",
                        help="Path to the DetectNet_v2 clustering config.")
    return parser.parse_args()


def main():
    """Running the inferencer client."""
    FLAGS = parse_command_line(sys.argv[1:])
    if FLAGS.mode.lower() == "detectnet_v2":
        assert os.path.isfile(FLAGS.postprocessing_config), (
            "Clustering config must be defined for DetectNet_v2."
        )
    log_level = "INFO"
    if FLAGS.verbose:
        log_level = "DEBUG"
    # Configure logging to get Maglev log messages.
    logging.basicConfig(format='%(asctime)s [%(levelname)s] '
                               '%(name)s: %(message)s',
                        level=log_level)

    if FLAGS.streaming and FLAGS.protocol.lower() != "grpc":
        raise Exception("Streaming is only allowed with gRPC protocol")

    try:
        if FLAGS.protocol.lower() == "grpc":
            # Create gRPC client for communicating with the server
            triton_client = grpcclient.InferenceServerClient(
                url=FLAGS.url, verbose=FLAGS.verbose)
        else:
            # Specify large enough concurrency to handle the
            # the number of requests.
            concurrency = 20 if FLAGS.async_set else 1
            triton_client = httpclient.InferenceServerClient(
                url=FLAGS.url, verbose=FLAGS.verbose, concurrency=concurrency)
    except Exception as e:
        print("client creation failed: " + str(e))
        sys.exit(1)

    # Make sure the model matches our requirements, and get some
    # properties of the model that we need for preprocessing
    try:
        model_metadata = triton_client.get_model_metadata(
            model_name=FLAGS.model_name, model_version=FLAGS.model_version)
    except InferenceServerException as e:
        print("failed to retrieve the metadata: " + str(e))
        sys.exit(1)

    try:
        model_config = triton_client.get_model_config(
            model_name=FLAGS.model_name, model_version=FLAGS.model_version)
    except InferenceServerException as e:
        print("failed to retrieve the config: " + str(e))
        sys.exit(1)

    if FLAGS.protocol.lower() == "grpc":
        model_config = model_config.config
    else:
        model_metadata, model_config = convert_http_metadata_config(
            model_metadata, model_config)

    triton_model = TRITON_MODEL_DICT[FLAGS.mode.lower()].from_metadata(model_metadata, model_config)
    target_shape = (triton_model.c, triton_model.h, triton_model.w)
    npdtype = triton_to_np_dtype(triton_model.triton_dtype)
    max_batch_size = triton_model.max_batch_size
    frames = []
    file_names = []

    if os.path.isdir(FLAGS.image_filename):
        frames = [
            Frame(os.path.join(FLAGS.image_filename, f),
                  triton_model.data_format,
                  npdtype,
                  target_shape)
            for f in os.listdir(FLAGS.image_filename)
            if os.path.isfile(os.path.join(FLAGS.image_filename, f)) and
            os.path.splitext(f)[-1] in [".jpg", ".jpeg", ".png"]
        ]

        for f in os.listdir(FLAGS.image_filename):
            if os.path.isfile(os.path.join(FLAGS.image_filename, f)) and \
                (os.path.splitext(f)[-1] in [".jpg", ".jpeg", ".png"]):
                    file_names.append(f.split(".")[0])
    else:
        frames = [
            Frame(os.path.join(FLAGS.image_filename),
                  triton_model.data_format,
                  npdtype,
                  target_shape)
        ]

    # Send requests of FLAGS.batch_size images. If the number of
    # images isn't an exact multiple of FLAGS.batch_size then just
    # start over with the first images until the batch is filled.
    requests = []
    responses = []
    result_filenames = []
    request_ids = []
    image_idx = 0
    last_request = False
    user_data = UserData()
    class_list = FLAGS.class_list.split(",")
    args_postprocessor = [
        FLAGS.batch_size, frames, FLAGS.output_path, triton_model.data_format
    ]
    if FLAGS.mode.lower() == "detectnet_v2":
        args_postprocessor.extend([class_list, FLAGS.postprocessing_config, target_shape])
    postprocessor = POSTPROCESSOR_DICT[FLAGS.mode.lower()](*args_postprocessor)

    # Holds the handles to the ongoing HTTP async requests.
    async_requests = []

    sent_count = 0

    if FLAGS.streaming:
        triton_client.start_stream(partial(completion_callback, user_data))

    logger.info("Sending inference request for batches of data")
    with tqdm(total=len(frames)) as pbar:
        while not last_request:
            input_filenames = []
            repeated_image_data = []

            for idx in range(FLAGS.batch_size):
                frame = frames[image_idx]
                if FLAGS.mode.lower() == "yolov3" or FLAGS.mode.lower() == "retinanet":
                    img = frame._load_img()
                    repeated_image_data.append(img)
                elif FLAGS.mode.lower() == "multitask_classification":
                    img = frame._load_img_multitask_classification()
                    repeated_image_data.append(img)
                elif FLAGS.mode.lower() == "peoplesegnet":
                    img = frame._load_img_maskrcnn()
                    repeated_image_data.append(img)
                else:
                    img = frame.load_image()
                    repeated_image_data.append(
                        triton_model.preprocess(
                            frame.as_numpy(img)
                        )
                    )
                image_idx = (image_idx + 1) % len(frames)
                if image_idx == 0:
                    last_request = True

            if max_batch_size > 0:
                batched_image_data = np.stack(repeated_image_data, axis=0)
            else:
                batched_image_data = repeated_image_data[0]

            # Send request
            try:
                req_gen_args = [batched_image_data, triton_model.input_names,
                    triton_model.output_names, triton_model.triton_dtype,
                    FLAGS.protocol.lower()]
                req_gen_kwargs = {}
                if FLAGS.mode.lower() == "classification":
                    req_gen_kwargs["num_classes"] = model_config.output[0].dims[0]
                req_generator = requestGenerator(*req_gen_args, **req_gen_kwargs)
                for inputs, outputs in req_generator:
                    sent_count += 1
                    if FLAGS.streaming:
                        triton_client.async_stream_infer(
                            FLAGS.model_name,
                            inputs,
                            request_id=str(sent_count),
                            model_version=FLAGS.model_version,
                            outputs=outputs)
                    elif FLAGS.async_set:
                        if FLAGS.protocol.lower() == "grpc":
                            triton_client.async_infer(
                                FLAGS.model_name,
                                inputs,
                                partial(completion_callback, user_data),
                                request_id=str(sent_count),
                                model_version=FLAGS.model_version,
                                outputs=outputs)
                        else:
                            async_requests.append(
                                triton_client.async_infer(
                                    FLAGS.model_name,
                                    inputs,
                                    request_id=str(sent_count),
                                    model_version=FLAGS.model_version,
                                    outputs=outputs))
                    else:
                        responses.append(
                            triton_client.infer(FLAGS.model_name,
                                                inputs,
                                                request_id=str(sent_count),
                                                model_version=FLAGS.model_version,
                                                outputs=outputs))

            except InferenceServerException as e:
                print("inference failed: " + str(e))
                if FLAGS.streaming:
                    triton_client.stop_stream()
                sys.exit(1)
            
            pbar.update(FLAGS.batch_size)

    if FLAGS.streaming:
        triton_client.stop_stream()

    if FLAGS.protocol.lower() == "grpc":
        if FLAGS.streaming or FLAGS.async_set:
            processed_count = 0
            while processed_count < sent_count:
                (results, error) = user_data._completed_requests.get()
                processed_count += 1
                if error is not None:
                    print("inference failed: " + str(error))
                    sys.exit(1)
                responses.append(results)
    else:
        if FLAGS.async_set:
            # Collect results from the ongoing async requests
            # for HTTP Async requests.
            for async_request in async_requests:
                responses.append(async_request.get_result())

    logger.info("Gathering responses from the server and post processing the inferenced outputs.")
    processed_request = 0

    # raw metrics 로깅을 위한 인스턴스 생성
    raw_metrics = RawMetrics()

    cnt = 0
    with tqdm(total=len(frames)) as pbar:
        while processed_request < sent_count:
            response = responses[processed_request]
            if FLAGS.protocol.lower() == "grpc":
                this_id = response.get_response().id
            else:
                this_id = response.get_response()["id"]

            postprocess_results = postprocessor.apply(
                response, this_id, render=True
            ) # (batch_size, N_predicted_object)

            for postprocess_result in postprocess_results["batchwise_boxes"]:
                pred_bboxes = []
                pred_labels = []
                for obj in postprocess_result:
                    pred_bboxes.append(obj.box)
                    pred_labels.append(obj.category)

                # print(file_names[cnt], pred_bboxes, pred_labels)

                try:
                    raw_metrics_res = raw_metrics.get_raw_metrics(
                        file_name=file_names[cnt],
                        pred_bboxes=pred_bboxes,
                        pred_labels=pred_labels,
                        nms_thr=0.45, 
                        score_thr=0.4, 
                        iou_thr=0.5
                    )
                    cnt += 1
                except:
                    pass
        
            processed_request += 1
            pbar.update(FLAGS.batch_size)
    
    logger.info("PASS")
    print(raw_metrics_res)

if __name__ == '__main__':
    main()
