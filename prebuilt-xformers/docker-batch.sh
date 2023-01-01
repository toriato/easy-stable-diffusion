#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail
shopt -s globstar
shopt -s nullglob
[[ "${TRACE-0}" == "1" ]] && set -o xtrace

cuda_versions=(
  "cu113"
  "cu116"
)

declare -A cuda_images
cuda_images["cu113"]="docker.io/nvidia/cuda:11.3.0-devel-ubuntu18.04"
cuda_images["cu116"]="docker.io/nvidia/cuda:11.6.0-devel-ubuntu18.04"

# xformers supports 3.7 to 3.10
# https://github.com/facebookresearch/xformers/blob/e163309908ed7a76847ce46c79b238b49fd7d341/setup.py#L335-L338
python_versions=(
  "3.7"
  "3.8"
  "3.9"
  "3.10"
)

for cuda_version in "${cuda_versions[@]}"; do
  dist_dir="wheels/${cuda_version}"
  mkdir -p "${dist_dir}"

  for python_version in "${python_versions[@]}"; do
    echo "build xformers with ${cuda_version} and python${python_version}"

    PYTHON_VERSION="${python_version}" \
    PIP_TORCH_INDEX="https://download.pytorch.org/whl/${cuda_version}" \
    PIP_TORCH_PACKAGE="torch==1.12.1+${cuda_version}" \
    DOCKER_IMAGE="${cuda_images["$cuda_version"]}" \
      "$(dirname "$0")/docker.sh"
    
    mv xformers/dist/*.whl "${dist_dir}/"
  done
done