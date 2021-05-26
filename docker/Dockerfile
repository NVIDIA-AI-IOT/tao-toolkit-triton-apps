FROM nvcr.io/nvidia/tritonserver:21.03-py3

# Setting up TensorRT Paths.
ENV TRT_LIB_PATH=/usr/lib/x86_64-linux-gnu
ENV TRT_INC_PATH=/usr/include/x86_64-linux-gnu

# Download and install TLT converter
RUN wget https://developer.nvidia.com/cuda111-cudnn80-trt72 -P /opt/tlt-converter && \
    apt-get update && apt-get install unzip libssl-dev -y && \
    unzip /opt/tlt-converter/cuda111-cudnn80-trt72 -d /opt/tlt-converter && \
    chmod +x /opt/tlt-converter/cuda11.1_cudnn8.0_trt7.2/tlt-converter

ENV PATH=/opt/tlt-converter/cuda11.1_cudnn8.0_trt7.2:$PATH

CMD ["/bin/bash"]
