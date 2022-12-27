# easy-stable-diffusion
[Open in Colab / 코랩에서 열기](https://colab.research.google.com/drive/1nBaePtwcW_ds7OQdFebcxB91n_aORQY5)

## prebuilt-xformers
다양한 Python 버전과 PyTorch 를 사용해 미리 컴파일한 [xformers](https://github.com/facebookresearch/xformers) 패키지

### 빌드 스크립트
`docker.sh` 로 개별 빌드하거나 `docker-batch.sh` 로 여러 버전 빌드할 수 있음

도커 (또는 포드만) 없어도 `build.sh` 로 도커 없이도 돌릴 수 있긴 함  
데비안 계열 배포판 안쓰고 있거나 잡다한 패키지 깔리는거 싫으면 도커 강추

#### 환경 변수
- `TRACE` - `1` 값으로 설정해 `xtrace` 켜기 (기본 값: `0`)
- `PYTHON_VERSION` - 사용할 파이썬 버전 (기본 값: `3.8`)
- `PIP_TORCH_INDEX` - PyTorch 패키지 인덱스 주소 (기본 값: `https://download.pytorch.org/whl/cu116`)
- `PIP_TORCH_PACKAGE` - PyTorch 패키지 명 (기본 값: `torch`)
- `TORCH_CUDA_ARCH_LIST` - `xformers` 에서 컴파일할 목표 GPU 버전 (기본 값: `6.0;6.1;6.2;7.0;7.2;7.5;8.0;8.6`, [버전 목록](https://en.wikipedia.org/wiki/CUDA#GPUs_supported))
- `NVCC_FLAGS` - NVCC 플래그 (기본 값: [`--use_fast_math -DXFORMERS_MEM_EFF_ATTENTION_DISABLE_BACKWARD`](https://github.com/facebookresearch/xformers/pull/482))
- `MAX_JOBS` - `ninja` 에서 사용할 스레드 수 (기본 값: `현재 스레드 수 - 2`)
- `DOCKER_IMAGE` - 빌드에 사용할 도커 이미지 (기본 값: `docker.io/nvidia/cuda:11.6.0-devel-ubuntu18.04`)

#### 예시
```sh
$ TRACE=1 PYTHON_VERSION="3.10" ./docker.sh && mv xformers/dist/*.whl .
```
