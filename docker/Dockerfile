FROM nvcr.io/nvidia/tritonserver:22.06-py3

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
RUN mkdir trt_oss_src && \
    cd trt_oss_src && \
    echo "$PWD Building TRT OSS..." && \
    git clone -b 22.02 https://github.com/nvidia/TensorRT TensorRT && \
    cd TensorRT && \
    git submodule update --init --recursive && \
    mkdir -p build && cd build  && \
    cmake .. -DGPU_ARCHS="35 52 53 60 61 70 75 80 86" -DTRT_LIB_DIR=/usr/lib/x86_64-linux-gnu -DTRT_BIN_DIR=`pwd`/out -DCUDA_VERSION=11.4 -DCUDNN_VERSION=8.4 && \
    make -j8 && \
    cp libnvinfer_plugin.so.8.2.* /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.8.2.5 && \
    cp libnvinfer_plugin_static.a /usr/lib/x86_64-linux-gnu/libnvinfer_plugin_static.a && \
    cp libnvonnxparser.so.8.2.* /usr/lib/x86_64-linux-gnu/libnvonnxparser.so.8.2.5 && \
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
