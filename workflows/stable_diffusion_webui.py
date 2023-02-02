import json
import shlex
from pathlib import Path

from modules import shared
from modules.log import Log
from modules.utils import mount_google_drive

log = Log()

try:
    from IPython.display import display
    display(log.widget)
except ImportError:
    pass


def main(**kwargs):
    with log:
        from .stable_diffusion_webui_modules import environment as env

        # 작업 경로 초기화
        workspace = Path(kwargs['workspace'])
        if shared.IN_COLAB and kwargs['use_google_drive']:
            workspace = mount_google_drive() / workspace

        override_file = workspace.joinpath('override.json')
        if override_file.is_file():
            with override_file.open() as file:
                override = json.load(file)

            for key, value in override.items():
                if key in env.options_ignore_override:
                    log.warn(f'override.json: "{key}" 값은 덮어쓸 수 없습니다.')
                    continue

                if key not in kwargs:
                    log.warn(f'override.json: "{key}" 값은 존재하지 않습니다.')
                    continue

                if type(value) != type(kwargs[key]):
                    log.warn(f'override.json: "{key}" 값의 자료형이 잘못됐습니다.')
                    continue

                log.info(f'override.json: "{key}" -> "{value}"')
                kwargs[key] = value

        env.options = env.Options(**{
            **kwargs,
            'workspace': Path(workspace),
            'args': shlex.split(kwargs['args']),
            'extra_args': shlex.split(kwargs['extra_args'])
        })

        log.print(str(env.options))
