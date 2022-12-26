#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail
shopt -s globstar
shopt -s nullglob
[[ "${TRACE-0}" == "1" ]] && set -o xtrace

# install requirements
apt update && apt install -y software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt install -y \
  git \
  python3-pip \
  python${PYTHON_VERSION} \
  python${PYTHON_VERSION}-venv \
  python${PYTHON_VERSION}-distutils

# venv is safer than update-alternatives
python${PYTHON_VERSION} -m venv /venv
source /venv/bin/activate

# checky check :)
pip -V

# setup xformers repository
if [[ ! -d "xformers/.git" ]]; then
  git clone --depth=1 https://github.com/facebookresearch/xformers.git
fi

pushd xformers

  # update repository
  git pull
  git submodule update --init --recursive

  # install python dependencies
  pip install \
    --upgrade \
    --extra-index-url "${PIP_TORCH_INDEX}" \
    "${PIP_TORCH_PACKAGE}" \
    ninja wheel \
    -r requirements.txt

  # just in case
  python setup.py clean

  # let's hope nothing bad happen...
  python setup.py build bdist_wheel --universal

popd
