ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.01-py3
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        libsndfile1-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY third_party/fairseq2 /opt/src/fairseq2

RUN python -m pip install --no-cache-dir \
        -r /opt/src/fairseq2/fairseq2n/python/requirements-build.txt \
    && cmake -GNinja \
        -S /opt/src/fairseq2/fairseq2n \
        -B /opt/src/fairseq2/fairseq2n/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTING=OFF \
        -DFAIRSEQ2N_USE_CUDA=OFF \
        -DFAIRSEQ2N_PYTHON_DEVEL=OFF \
    && cmake --build /opt/src/fairseq2/fairseq2n/build --parallel 8 \
    && python -m pip install --no-cache-dir --no-build-isolation \
        /opt/src/fairseq2/fairseq2n/python \
    && python -m pip install --no-cache-dir --no-build-isolation \
        /opt/src/fairseq2

COPY third_party/torchaudio /opt/src/torchaudio

RUN BUILD_SOX=0 \
    USE_FFMPEG=0 \
    USE_CUDA=0 \
    BUILD_RIR=0 \
    BUILD_RNNT=0 \
    BUILD_ALIGN=0 \
    BUILD_CUDA_CTC_DECODER=0 \
    CMAKE_BUILD_PARALLEL_LEVEL=8 \
    python -m pip install --no-cache-dir --no-build-isolation --no-deps \
        /opt/src/torchaudio

COPY third_party/seamless_communication /opt/src/seamless_communication
COPY config/s0-python.constraints.txt /opt/streamglass/s0-python.constraints.txt

RUN python -m pip install --no-cache-dir --no-build-isolation \
        --constraint /opt/streamglass/s0-python.constraints.txt \
        /opt/src/seamless_communication

COPY deploy/s0/verify_runtime.py /usr/local/bin/streamglass-verify-s0

ENTRYPOINT []
CMD ["python", "/usr/local/bin/streamglass-verify-s0"]
