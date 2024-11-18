# Stage 1: Build the required conda environment
FROM continuumio/miniconda3:latest as condainstall

ARG PYTHON=3.10
ARG CUDA_VERSION=11.8
ARG PYTORCH=2.1.2
ARG OPENCLIP=2.23

RUN --mount=type=cache,target=/opt/conda/pkgs \
    conda create --copy -p /env python="$PYTHON" && \
    conda install -p /env -c pytorch -c nvidia pytorch="$PYTORCH" torchvision pytorch-cuda="$CUDA_VERSION" && \
    conda install -p /env -c conda-forge hydra-core transformers accelerate optimum ftfy timm sentencepiece tqdm wandb unidecode tabulate && \
    conda run -p /env python -m pip install --no-cache-dir openai opencv-python && \
    conda run -p /env python -m pip install --no-cache-dir --no-deps open_clip_torch=="$OPENCLIP" git+https://github.com/openai/CLIP.git && \
    conda run -p /env python -m pip check && sleep 3s

# Work around a warning when using image embedder for PyTorch 2 and CUDA 11.8 (https://stackoverflow.com/questions/76216778/userwarning-applied-workaround-for-cudnn-issue-install-nvrtc-so, https://github.com/pytorch/pytorch/issues/97041)
RUN [[ "$CUDA_VERSION" == "11.8" ]] && ln -sn libnvrtc.so.11.2 /env/lib/libnvrtc.so || true

RUN find -name '*.a' -delete && \
    find -name '__pycache__' -type d -exec rm -rf '{}' '+'

# Stage 2: Repackage into a small Ubuntu image
FROM ubuntu:22.04

ENV PATH="/env/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*

COPY --from=condainstall /env /env
# EOF
