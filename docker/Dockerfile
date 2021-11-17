FROM nvcr.io/nvidia/tritonserver:21.10-py3

# Setting up TensorRT Paths.
ENV TRT_LIB_PATH=/usr/lib/x86_64-linux-gnu
ENV TRT_INC_PATH=/usr/include/x86_64-linux-gnu

# Download and install TAO Toolkit converter
RUN wget https://developer.nvidia.com/tao-converter-80 -P /opt/tao-converter && \
    apt-get update && apt-get install unzip libssl-dev -y && \
    unzip /opt/tao-converter/tao-converter-80 -d /opt/tao-converter && \
    chmod +x /opt/tao-converter/cuda11.3-trt8.0/tao-converter

ENV PATH=/opt/tao-converter/cuda11.3-trt8.0:$PATH

CMD ["/bin/bash"]
