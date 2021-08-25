FROM nvcr.io/nvidia/tritonserver:21.03-py3

# Setting up TensorRT Paths.
ENV TRT_LIB_PATH=/usr/lib/x86_64-linux-gnu
ENV TRT_INC_PATH=/usr/include/x86_64-linux-gnu

# Download and install TAO Toolkit converter
RUN wget https://developer.nvidia.com/cuda111-cudnn80-trt72-0 -P /opt/tao-converter && \
    apt-get update && apt-get install unzip libssl-dev -y && \
    unzip /opt/tao-converter/cuda111-cudnn80-trt72-0 -d /opt/tao-converter && \
    chmod +x /opt/tao-converter/cuda11.1-trt7.2/tao-converter

ENV PATH=/opt/tao-converter/cuda11.1-trt7.2:$PATH

CMD ["/bin/bash"]
