FROM nvcr.io/nvidia/tritonserver:23.02-py3

# Installing TRT OSS to the Triton Container.
RUN apt-get update && \
    apt-get install -y pkg-config && \
    apt-get install -y git && \
    apt-get install -y zlib1g && \
    apt-get install -y zlib1g-dev
RUN cd /tmp \
    && wget https://github.com/Kitware/CMake/releases/download/v3.14.4/cmake-3.14.4-Linux-x86_64.sh  \
    && chmod +x cmake-3.14.4-Linux-x86_64.sh  \
    && ./cmake-3.14.4-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license  \
    && rm ./cmake-3.14.4-Linux-x86_64.sh \
    && cd -
RUN cd /opt
RUN ln -s /usr/bin/python3 /usr/bin/python
# Clone and checkout TensorRT OSS
# Moving TensorRT to 8.5 branch.
ENV TRT_TAG "release/8.5"
ENV TRT_INCLUDE_DIR="/usr/include/x86_64-linux-gnu"
# Install TRT OSS
RUN mkdir trt_oss_src && \
    cd trt_oss_src && \
    echo "$PWD Building TRT OSS..." && \
    git clone -b $TRT_TAG https://github.com/NVIDIA/TensorRT.git TensorRT && \
    cd TensorRT && \
    git submodule update --init --recursive && \
    mkdir -p build && cd build  && \
    cmake .. -DGPU_ARCHS="53 60 61 70 75 80 86 90" -DTRT_LIB_DIR=/usr/lib/x86_64-linux-gnu -DTRT_BIN_DIR=`pwd`/out -DCUDA_VERSION=11.8 -DCUDNN_VERSION=8.7 && \
    make -j32 && \
    cp libnvinfer_plugin.so.8.5.3 /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.8.5.3 && \
    cp libnvinfer_plugin_static.a /usr/lib/x86_64-linux-gnu/libnvinfer_plugin_static.a && \
    cp libnvonnxparser.so.8.5.3 /usr/lib/x86_64-linux-gnu/libnvonnxparser.so.8.5.3 && \
    cp libnvcaffeparser.so.8.5.3 /usr/lib/x86_64-linux-gnu/libnvcaffeparser.so.8.5.3 && \
    cp trtexec /usr/local/bin/ && \
    cd ../../../ && \
    rm -rf trt_oss_src

# Setting up TensorRT Paths.
ENV TRT_LIB_PATH=/usr/lib/x86_64-linux-gnu
ENV TRT_INC_PATH=/usr/include/x86_64-linux-gnu

# Download and install TAO Toolkit converter
RUN mkdir /opt/tao-converter && \
    wget --content-disposition https://api.ngc.nvidia.com/v2/resources/nvidia/tao/tao-converter/versions/v3.22.05_trt8.2_x86/zip -O /opt/tao-converter/tao-converter_v3.22.05_trt8.2_x86.zip && \
    apt update && apt install unzip libssl-dev -y && \
    unzip /opt/tao-converter/tao-converter_v3.22.05_trt8.2_x86.zip -d /opt/tao-converter && \
    chmod +x /opt/tao-converter/tao-converter && \
    rm -rf /opt/tao-converter/tao-converter_v3.22.05_trt8.2_x86.zip

ENV PATH=/opt/tao-converter:$PATH

CMD ["/bin/bash"]
