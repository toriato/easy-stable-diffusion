#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail
shopt -s globstar
shopt -s nullglob
[[ "${TRACE-0}" == "1" ]] && set -o xtrace
 
docker run \
  -e TRACE="${TRACE-0}" \
  -e PYTHON_VERSION="${PYTHON_VERSION:-"3.8"}" \
  -e PIP_TORCH_INDEX="${PIP_TORCH_INDEX:-"https://download.pytorch.org/whl/cu116"}" \
  -e PIP_TORCH_PACKAGE="${PIP_TORCH_PACKAGE:-"torch"}" \
  -e TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-"6.0;6.1;6.2;7.0;7.2;7.5;8.0;8.6"}" \
  -e NVCC_FLAGS="${NVCC_FLAGS:-"--use_fast_math -DXFORMERS_MEM_EFF_ATTENTION_DISABLE_BACKWARD"}" \
  -e MAX_JOBS="${MAX_JOBS:-$(($(nproc) - 2))}" \
  -e FORCE_CUDA=1 \
  -e XFORMERS_DISABLE_FLASH_ATTN=1 \
  -e DEBIAN_FRONTEND=noninteractive \
  -v "$(dirname "$0")/build.sh:/build.sh:ro" \
  -v "$(pwd):/workspace" -w /workspace \
  --entrypoint /build.sh \
  "${DOCKER_IMAGE:-"docker.io/nvidia/cuda:11.6.0-devel-ubuntu18.04"}"
