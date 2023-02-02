import json
import shlex
from pathlib import Path
from typing import List, NamedTuple

from modules import shared
from modules.log import Log
from modules.subprocess import call, call_python
from modules.utils import hook_runtime_disconnect, mount_google_drive


class Options(NamedTuple):
    workspace: Path
    disconnect_runtime: bool
    use_google_drive: bool
    use_xformers: bool
    use_gradio: bool
    gradio_username: str
    gradio_password: str
    ngrok_api_token: str
    python_executable: str
    repo_url: str
    repo_commit: str
    args: List[str]
    extra_args: List[str]


options: Options
options_ignore_override = [
    'workspace'
]


def setup_options(**kwargs):
    global options

    workspace = Path(kwargs['workspace'])

    if shared.IN_COLAB:
        # 작업 경로 초기화
        if kwargs['use_google_drive']:
            workspace = mount_google_drive() / 'MyDrive' / workspace
            assert workspace.parent.is_dir(), '구글 드라이브 마운팅에 실패했습니다!'

        # 런타임 자동 해제
        if kwargs['disconnect_runtime']:
            hook_runtime_disconnect()

    override_file = workspace.joinpath('override.json')
    if override_file.is_file():
        with Log.info('override.json 파일이 존재합니다, 설정 값을 덮어씁니다.'):
            with override_file.open() as file:
                override = json.load(file)

            for key, value in override.items():
                if key in options_ignore_override:
                    Log.warning(f'"{key}" 값은 덮어쓸 수 없습니다.')
                elif key not in kwargs:
                    Log.warning(f'"{key}" 값은 존재하지 않습니다.')
                elif type(value) != type(kwargs[key]):
                    Log.warning(f'"{key}" 값의 자료형이 잘못됐습니다.')
                else:
                    Log.info(f'"{key}" -> "{value}"')
                    kwargs[key] = value

    options = Options(**{
        **kwargs,
        'workspace': Path(workspace),
        'args': shlex.split(kwargs['args']),
        'extra_args': shlex.split(kwargs['extra_args'])
    })


def setup_repository():
    path = options.workspace / 'stable-diffusion-webui'

    with Log.info('저장소를 설정합니다.'):
        if not path.joinpath('.git').is_dir():
            call(['git', 'clone', options.repo_url, str(path)])

        if options.repo_commit:
            call(['git', 'stash'], cwd=path)
            call(['git', 'checkout', options.repo_commit], cwd=path)


def setup_dependencies():
    pass
