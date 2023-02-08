import os
from shutil import rmtree
from subprocess import call
from tempfile import mkdtemp
from typing import Callable

from modules import sd_models, shared

load_model_weights: Callable


def alternate_load_model_weights(model, checkpoint_info: sd_models.CheckpointInfo, *args, **kwargs):
    # 기본적으로 모델을 전부 불러오기 전까진 변경된 설정이 저장되지 않음
    # 불러오는 중 VRAM 부족 등으로 프로세스가 종료되면 다음 실행 때 이전 모델을 불러오게 됨
    shared.opts.save(shared.config_filename)

    print('Copying model into temporary directory.')

    # rsync 로 사용자에게 모델 복사까지 남은 시간 보여주기
    temp_dir = mkdtemp()
    copied_checkpoint_file = os.path.join(temp_dir, checkpoint_info.name)
    call(['rsync', '-aP', checkpoint_info.filename, copied_checkpoint_file])

    print(f'Successfully copied model to {copied_checkpoint_file}')

    try:
        sd = load_model_weights(
            model,
            sd_models.CheckpointInfo(copied_checkpoint_file),
            *args, **kwargs
        )
    finally:
        print('Discarding temporary model file.')
        rmtree(temp_dir, True)

    return sd


if not sd_models.load_model_weights == alternate_load_model_weights:
    print('Applying alternate load_model_weights.')
    load_model_weights = sd_models.load_model_weights
    sd_models.load_model_weights = alternate_load_model_weights
