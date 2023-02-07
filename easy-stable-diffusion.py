import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from distutils.spawn import find_executable
from importlib.util import find_spec
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import requests
import torch

OPTIONS = {}

# fmt: off
#####################################################
# 코랩 노트북에선 #@param 문법으로 사용자로부터 설정 값을 가져올 수 있음
# 다른 환경일 땐 override.json 파일 등을 사용해야함
#####################################################
#@title

#@markdown ### <font color="orange">***작업 디렉터리 경로***</font>
#@markdown 임베딩, 모델, 결과와 설정 파일 등이 영구적으로 보관될 디렉터리 경로
WORKSPACE = 'SD' #@param {type:"string"}

#@markdown ##### <font color="orange">***자동으로 코랩 런타임을 종료할지?***</font>
DISCONNECT_RUNTIME = True  #@param {type:"boolean"}
OPTIONS['DISCONNECT_RUNTIME'] = DISCONNECT_RUNTIME

#@markdown ##### <font color="orange">***구글 드라이브와 동기화할지?***</font>
#@markdown <font color="red">**주의**</font>: 동기화 전 남은 용량이 충분한지 확인 필수 (5GB 이상)
USE_GOOGLE_DRIVE = True  #@param {type:"boolean"}
OPTIONS['USE_GOOGLE_DRIVE'] = USE_GOOGLE_DRIVE

#@markdown ##### <font color="orange">***xformers 를 사용할지?***</font>
#@markdown - <font color="green">장점</font>: 이미지 생성 속도 개선 가능성 있음
#@markdown - <font color="red">단점</font>: 출력한 그림의 질이 조금 떨어질 수 있음
USE_XFORMERS = True  #@param {type:"boolean"}
OPTIONS['USE_XFORMERS'] = USE_XFORMERS

#@markdown ##### <font color="orange">***Gradio 터널을 사용할지?***</font>
#@markdown - <font color="green">장점</font>: 따로 설정할 필요가 없어 편리함
#@markdown - <font color="red">**단점**</font>: 접속이 느리고 끊기거나 버튼이 안 눌리는 등 오류 빈도가 높음
USE_GRADIO = True #@param {type:"boolean"}
OPTIONS['USE_GRADIO'] = USE_GRADIO

#@markdown ##### <font color="orange">***Gradio 인증 정보***</font>
#@markdown Gradio 접속 시 사용할 사용자 아이디와 비밀번호
#@markdown <br>`GRADIO_USERNAME` 입력 란에 `user1:pass1,user,pass2`처럼 입력하면 여러 사용자 추가 가능
#@markdown <br>`GRADIO_USERNAME` 입력 란을 <font color="red">비워두면</font> 인증 과정을 사용하지 않음
#@markdown <br>`GRADIO_PASSWORD` 입력 란을 <font color="red">비워두면</font> 자동으로 비밀번호를 생성함
GRADIO_USERNAME = 'gradio' #@param {type:"string"}
GRADIO_PASSWORD = '' #@param {type:"string"}
GRADIO_PASSWORD_GENERATED = False
OPTIONS['GRADIO_USERNAME'] = GRADIO_USERNAME
OPTIONS['GRADIO_PASSWORD'] = GRADIO_PASSWORD

#@markdown ##### <font color="orange">***ngrok API 키***</font>
#@markdown ngrok 터널에 사용할 API 토큰
#@markdown <br>[설정하는 방법은 여기를 클릭해 확인](https://arca.live/b/aiart/60683088), [API 토큰은 여기를 눌러 계정을 만든 뒤 얻을 수 있음](https://dashboard.ngrok.com/get-started/your-authtoken)
#@markdown <br>입력 란을 <font color="red">비워두면</font> ngrok 터널을 비활성화함
#@markdown - <font color="green">장점</font>: 접속이 빠른 편이고 타임아웃이 거의 발생하지 않음
#@markdown - <font color="red">**단점**</font>: 계정을 만들고 API 토큰을 직접 입력해줘야함
NGROK_API_TOKEN = '' #@param {type:"string"}
NGROK_URL = None
OPTIONS['NGROK_API_TOKEN'] = NGROK_API_TOKEN

#@markdown ##### <font color="orange">***WebUI 레포지토리 주소***</font>
REPO_URL = 'https://github.com/AUTOMATIC1111/stable-diffusion-webui.git' #@param {type:"string"}
OPTIONS['REPO_URL'] = REPO_URL

#@markdown ##### <font color="orange">***WebUI 레포지토리 커밋 해시***</font>
#@markdown 업데이트가 실시간으로 올라올 때 최신 버전에서 오류가 발생할 때 [레포지토리 커밋 목록](https://github.com/AUTOMATIC1111/stable-diffusion-webui/commits/master)에서
#@markdown <br>과거 커밋 해시 값[(영문과 숫자로된 난수 값; 예시 이미지)](https://vmm.pw/MzMy)을 아래에 붙여넣은 뒤 실행하면 과거 버전을 사용할 수 있음
#@markdown <br>입력 란을 <font color="red">비워두면</font> 가장 최신 커밋을 가져옴
REPO_COMMIT = '' #@param {type:"string"}
OPTIONS['REPO_COMMIT'] = REPO_COMMIT

#@markdown ##### <font color="orange">***Python 바이너리 이름***</font>
#@markdown 입력 란을 <font color="red">비워두면</font> 시스템에 설치된 Python 을 사용함
PYTHON_EXECUTABLE = '' #@param {type:"string"}
OPTIONS['PYTHON_EXECUTABLE'] = PYTHON_EXECUTABLE

#@markdown ##### <font color="orange">***WebUI 인자***</font>
#@markdown <font color="red">**주의**</font>: 비어있지 않으면 실행에 필요한 인자가 자동으로 생성되지 않음
#@markdown <br>[사용할 수 있는 인자 목록](https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/shared.py#L23)
ARGS = '' #@param {type:"string"}
OPTIONS['ARGS'] = shlex.split(ARGS)

#@markdown ##### <font color="orange">***WebUI 추가 인자***</font>
EXTRA_ARGS = '' #@param {type:"string"}
OPTIONS['EXTRA_ARGS'] = shlex.split(EXTRA_ARGS)

#####################################################
# 사용자 설정 값 끝
#####################################################
# fmt: on

# 로그 변수
LOG_FILE: Optional[io.TextIOWrapper] = None
LOG_WIDGET = None
LOG_BLOCKS = []

# 로그 HTML 위젯 스타일
LOG_WIDGET_STYLES = {
    'wrapper': {
        'overflow-x': 'auto',
        'max-width': '100%',
        'padding': '1em',
        'background-color': 'black',
        'white-space': 'pre',
        'font-family': 'monospace',
        'font-size': '1em',
        'line-height': '1.1em',
        'color': 'white'
    },
    'dialog': {
        'display': 'block',
        'margin-top': '.5em',
        'padding': '.5em',
        'font-weight': 'bold',
        'font-size': '1.5em',
        'line-height': '1em',
        'color': 'black'
    }
}
LOG_WIDGET_STYLES['dialog_success'] = {
    **LOG_WIDGET_STYLES['dialog'],
    'border': '3px dashed darkgreen',
    'background-color': 'green',
}
LOG_WIDGET_STYLES['dialog_warning'] = {
    **LOG_WIDGET_STYLES['dialog'],
    'border': '3px dashed darkyellow',
    'background-color': 'yellow',
}
LOG_WIDGET_STYLES['dialog_error'] = {
    **LOG_WIDGET_STYLES['dialog'],
    'border': '3px dashed darkred',
    'background-color': 'red',
}

IN_INTERACTIVE = hasattr(sys, 'ps1')
IN_COLAB = False

try:
    from IPython import get_ipython
    IN_COLAB = 'google.colab' in str(get_ipython())
except ImportError:
    pass


def hook_runtime_disconnect():
    if not IN_COLAB:
        return

    from google.colab import runtime

    # asyncio 는 여러 겹으로 사용할 수 없게끔 설계됐기 때문에
    # 주피터 노트북 등 이미 루프가 돌고 있는 곳에선 사용할 수 없음
    # 이는 nest-asyncio 패키지를 통해 어느정도 우회하여 사용할 수 있음
    # https://pypi.org/project/nest-asyncio/
    if not has_python_package('nest_asyncio'):
        execute(['pip', 'install', 'nest-asyncio'])

    import nest_asyncio
    nest_asyncio.apply()

    import asyncio

    async def unassign():
        runtime.unassign()

    # 평범한 환경에선 비동기로 동작하여 바로 실행되나
    # 코랩 런타임에선 순차적으로 실행되기 때문에 현재 셀 종료 후 즉시 실행됨
    asyncio.create_task(unassign())


def setup_colab():
    global WORKSPACE

    # 구글 드라이브 마운트하기
    if OPTIONS['USE_GOOGLE_DRIVE']:
        from google.colab import drive
        drive.mount('drive')

        WORKSPACE = str(
            Path('drive', 'MyDrive').joinpath(WORKSPACE)
        )

    if not OPTIONS['USE_GRADIO'] and not OPTIONS['NGROK_API_TOKEN']:
        alert('선택된 터널링 서비스가 없습니다!')

    if OPTIONS['PYTHON_EXECUTABLE'] and not find_executable(OPTIONS['PYTHON_EXECUTABLE']):
        execute(['apt', 'install', OPTIONS['PYTHON_EXECUTABLE']])
        execute(
            f"curl -sS https://bootstrap.pypa.io/get-pip.py | {OPTIONS['PYTHON_EXECUTABLE']}")

    if len(OPTIONS['ARGS']) < 1:
        # --listen 또는 --share 인자를 사용하면 확장 기능 탭이 막혀버림
        # 어처피 Gradio 비밀번호는 자동으로 생성되는데
        # 일부러 제거하고 외부 접근 공개한 바보 책임이니 인자 넣어둠
        OPTIONS['EXTRA_ARGS'] += ['--enable-insecure-extension-access']

        if not torch.cuda.is_available():
            alert('GPU 런타임이 아닙니다, 할당량이 초과 됐을 수도 있습니다!')

            OPTIONS['EXTRA_ARGS'] += [
                '--skip-torch-cuda-test',
                '--no-half',
                '--opt-sub-quad-attention'
            ]


def setup_environment():
    global LOG_WIDGET

    # 노트북 환경이라면 로그 표시를 위한 HTML 요소 만들기
    if IN_INTERACTIVE:
        try:
            from IPython.display import display
            from ipywidgets import widgets

            LOG_WIDGET = widgets.HTML()
            display(LOG_WIDGET)

        except ImportError:
            pass

    # google.colab 패키지가 있다면 코랩 환경으로 인식하기
    if IN_COLAB:
        setup_colab()

    # 로그 파일 만들기
    global LOG_FILE
    workspace = Path(WORKSPACE)
    log_path = workspace.joinpath(
        'logs',
        datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M-%S.log')
    )

    log_path.parent.mkdir(0o777, True, True)

    LOG_FILE = log_path.open('a')

    # 현재 환경 출력
    import platform
    log(' '.join(os.uname()))
    log(f'Python {platform.python_version()}')
    log(str(Path().resolve()))

    # 덮어쓸 설정 파일 가져오기
    override_path = workspace.joinpath('override.json')
    if override_path.exists():
        with override_path.open('r') as file:
            override_options = json.loads(file.read())
            for key, value in override_options.items():
                if key not in OPTIONS:
                    log(f'{key} 키는 존재하지 않는 설정입니다', styles={'color': 'red'})
                    continue

                if type(value) != type(OPTIONS[key]):
                    log(f'{key} 키는 {type(OPTIONS[key]).__name__} 자료형이여만 합니다', styles={
                        'color': 'red'})
                    continue

                OPTIONS[key] = value

                log(f'override.json: {key} = {json.dumps(value)}')

    # 체크포인트 모델이 존재하지 않는다면 기본 모델 받아오기
    if not has_checkpoint():
        for file in [
            {
                'url': 'https://huggingface.co/saltacc/wd-1-4-anime/resolve/main/wd-1-4-epoch2-fp16.safetensors',
                'target': os.path.join(WORKSPACE, 'models/Stable-diffusion/wd-1-4-epoch2-fp16.safetensors'),
                'summary': '기본 체크포인트 파일을 받아옵니다'
            },
            {
                'url': 'https://huggingface.co/saltacc/wd-1-4-anime/resolve/main/wd-1-4-epoch2-fp16.yaml',
                'target': os.path.join(WORKSPACE, 'models/Stable-diffusion/wd-1-4-epoch2-fp16.yaml'),
                'summary': '기본 체크포인트 설정 파일을 받아옵니다'
            },
            {
                'url': 'https://huggingface.co/saltacc/wd-1-4-anime/resolve/main/VAE/kl-f8-anime2.ckpt',
                'target': os.path.join(WORKSPACE, 'models/VAE/kl-f8-anime2.ckpt'),
                'summary': '기본 VAE 파일을 받아옵니다'
            }
        ]:
            download(**file)


# ==============================
# 로그
# ==============================


def format_styles(styles: dict) -> str:
    return ';'.join(map(lambda kv: ':'.join(kv), styles.items()))


def format_list(value):
    if isinstance(value, dict):
        return '\n'.join(map(lambda kv: f'{kv[0]}: {kv[1]}', value.items()))
    else:
        return '\n'.join(value)


def render_log() -> None:
    try:
        from ipywidgets import widgets
    except ImportError:
        return

    if not isinstance(LOG_WIDGET, widgets.HTML):
        return

    html = f'''<div style="{format_styles(LOG_WIDGET_STYLES['wrapper'])}">'''

    for block in LOG_BLOCKS:
        styles = {
            'display': 'inline-block',
            **block['styles']
        }
        child_styles = {
            'display': 'inline-block',
            **block['child_styles']
        }

        html += f'<span style="{format_styles(styles)}">{block["msg"]}</span>\n'

        if block['max_childs'] is not None and len(block['childs']) > 0:
            html += f'<div style="{format_styles(child_styles)}">'
            html += ''.join(block['childs'][-block['max_childs']:])
            html += '</div>'

    html += '</div>'

    LOG_WIDGET.value = html


def log(
    msg: str,
    styles={},
    newline=True,

    parent=False,
    parent_index: Optional[int] = None,
    child_styles={
        'padding-left': '1em',
        'color': 'gray'
    },
    max_childs=0,

    print_to_file=True,
    print_to_widget=True
) -> Optional[int]:
    # 기록할 내용이 ngrok API 키와 일치한다면 숨기기
    # TODO: 더 나은 문자열 검사, 원치 않은 내용이 가려질 수도 있음
    if OPTIONS['NGROK_API_TOKEN'] != '':
        msg = msg.replace(OPTIONS['NGROK_API_TOKEN'], '**REDACTED**')

    if newline:
        msg += '\n'

    # 파일에 기록하기
    if print_to_file and LOG_FILE:
        if parent_index and msg.endswith('\n'):
            LOG_FILE.write('\t')
        LOG_FILE.write(msg)
        LOG_FILE.flush()

    # 로그 위젯에 기록하기
    if print_to_widget and LOG_WIDGET:
        # 부모 로그가 없다면 새 블록 만들기
        if parent or parent_index is None:
            LOG_BLOCKS.append({
                'msg': msg,
                'styles': styles,
                'childs': [],
                'child_styles': child_styles,
                'max_childs': max_childs
            })
            render_log()
            return len(LOG_BLOCKS) - 1

        # 부모 로그가 존재한다면 추가하기
        LOG_BLOCKS[parent_index]['childs'].append(msg)
        render_log()

    print('\t' if parent_index else '' + msg, end='')


def log_trace() -> None:
    import sys
    import traceback

    # 스택 가져오기
    ex_type, ex_value, ex_traceback = sys.exc_info()

    styles = {}

    # 오류가 존재한다면 메세지 빨간색으로 출력하기
    # https://docs.python.org/3/library/sys.html#sys.exc_info
    # TODO: 오류 유무 이렇게 확인하면 안될거 같은데 일단 귀찮아서 대충 써둠
    if ex_type is not None:
        styles = LOG_WIDGET_STYLES['dialog_error']

    parent_index = log(
        '오류가 발생했습니다, <a href="https://discord.gg/6wQeA2QXgM">디스코드 서버</a>에 보고해주세요',
        styles)
    assert parent_index

    # 오류가 존재한다면 오류 정보와 스택 트레이스 출력하기
    if ex_type is not None:
        log(f'{ex_type.__name__}: {ex_value}', parent_index=parent_index)
        log(
            format_list(
                map(
                    lambda v: f'{v[0]}#{v[1]}\n\t{v[2]}\n\t{v[3]}',
                    traceback.extract_tb(ex_traceback))
            ),
            parent_index=parent_index
        )

    # 로그 파일이 없으면 보고하지 않기
    # TODO: 로그 파일이 존재하지 않을 수가 있나...?
    if not LOG_FILE:
        log('로그 파일이 존재하지 않습니다, 보고서를 만들지 않습니다')
        return

    # 로그 위젯이 존재한다면 보고서 올리고 내용 업데이트하기
    if LOG_WIDGET:
        # 이전 로그 전부 긁어오기
        logs = ''
        with open(LOG_FILE.name) as file:
            logs = file.read()

        # 로그 업로드
        # TODO: 업로드 실패 시 오류 처리
        res = requests.post(
            'https://hastebin.com/documents',
            data=logs.encode('utf-8')
        )
        url = f"https://hastebin.com/raw/{json.loads(res.text)['key']}"

        # 기존 오류 메세지 업데이트
        LOG_BLOCKS[parent_index]['msg'] = '\n'.join([
            '오류가 발생했습니다, 아래 주소를 <a href="https://discord.gg/6wQeA2QXgM">디스코드 서버</a>에 보고해주세요',
            f'<a target="_blank" href="{url}">{url}</a>',
        ])

        render_log()


def alert(message: str):
    log(message)

    if not IN_INTERACTIVE:
        return

    from IPython.display import display
    from ipywidgets import widgets

    display(
        widgets.HTML(f'<script>alert({json.dumps(message)})</script>')
    )


# ==============================
# 서브 프로세스
# ==============================
def execute(
    args: Union[str, List[str]],
    parser: Optional[
        Callable[[str, Optional[int]], None]
    ] = None,
    summary: Optional[str] = None,
    hide_summary=False,
    print_to_file=True,
    print_to_widget=True,
    throw=True,
    **kwargs
) -> Tuple[str, int]:
    if isinstance(args, str) and 'shell' not in kwargs:
        kwargs['shell'] = True

    # 서브 프로세스 만들기
    p = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding='utf-8',
        **kwargs)

    # 로그에 시작한 프로세스 정보 출력하기
    formatted_args = args if isinstance(args, str) else ' '.join(args)
    summary = formatted_args if summary is None else f'{summary}\n   {formatted_args}'

    log_index = log(
        f'=> {summary}',
        styles={'color': 'yellow'},
        max_childs=5)

    output = ''

    # 프로세스 출력 위젯에 리다이렉션하기
    while p.poll() is None:
        # 출력이 비어있다면 넘어가기
        assert p.stdout
        line = p.stdout.readline()
        if not line:
            continue

        # 프로세스 출력 버퍼에 추가하기
        output += line

        # 파서 함수 실행하기
        if callable(parser):
            try:
                if parser(line, log_index):
                    continue
            except:
                log_trace()

        # 프로세스 출력 로그하기
        log(
            line,
            newline=False,
            parent_index=log_index,
            print_to_file=print_to_file,
            print_to_widget=print_to_widget)

    # 변수 정리하기
    rc = p.poll()
    assert rc is not None

    # 로그 블록 업데이트
    if LOG_WIDGET:
        assert log_index

        if rc == 0:
            # 성공적으로 프로세스가 종료됐을 때
            if hide_summary:
                # 현재 로그 블록 숨기기 (제거하기)
                del LOG_BLOCKS[log_index]
            else:
                # 현재 로그 텍스트 초록색으로 변경하고 프로세스 출력 숨기기
                LOG_BLOCKS[log_index]['styles']['color'] = 'green'
                LOG_BLOCKS[log_index]['max_childs'] = None
        else:
            # 현재 로그 텍스트 빨간색으로 변경하고 프로세스 출력 모두 표시하기
            LOG_BLOCKS[log_index]['styles']['color'] = 'red'
            LOG_BLOCKS[log_index]['max_childs'] = 0

        # 로그 블록 렌더링
        render_log()

    # 오류 코드를 반환했다면
    if rc != 0 and throw:
        raise Exception(f'프로세스가 {rc} 코드를 반환했습니다')

    return output, rc

# ==============================
# 작업 경로
# ==============================


def delete(path: os.PathLike) -> None:
    path = Path(path)

    if path.is_file() or path.is_symlink():
        path.unlink()
    else:
        shutil.rmtree(path, ignore_errors=True)


def has_python_package(pkg: str, executable: Optional[str] = None) -> bool:
    if not executable:
        return find_spec(pkg) is not None

    _, rc = execute(
        [
            executable, '-c',
            f'''
            import importlib
            import sys
            sys.exit(0 if importlib.find_loader({shlex.quote(pkg)}) else 0)
            '''
        ],
        throw=False)

    return True if rc == 0 else False


# ==============================
# 파일 다운로드
# ==============================
def download(url: str, target: str, ignore_aria2=False, **kwargs):
    # 파일을 받을 디렉터리 만들기
    Path(target).parent.mkdir(0o777, True, True)

    # 빠른 다운로드를 위해 aria2 패키지 설치 시도하기
    if not ignore_aria2:
        if not find_executable('aria2c') and find_executable('apt'):
            execute(['apt', 'install', 'aria2'])

        if find_executable('aria2c'):
            execute(
                [
                    'aria2c',
                    '--continue',
                    '--always-resume',
                    '--summary-interval', '10',
                    '--disk-cache', '64M',
                    '--min-split-size', '8M',
                    '--max-concurrent-downloads', '8',
                    '--max-connection-per-server', '8',
                    '--max-overall-download-limit', '0',
                    '--max-download-limit', '0',
                    '--split', '8',
                    '--out', target,
                    url
                ],
                **kwargs)

    elif find_executable('curl'):
        execute(
            [
                'curl',
                '--location',
                '--output', target,
                url
            ],
            **kwargs)

    else:
        if 'summary' in kwargs.keys():
            log(kwargs.pop('summary'), **kwargs)

        with requests.get(url, stream=True) as res:
            res.raise_for_status()

            with open(target, 'wb') as file:
                # 받아온 파일 디코딩하기
                # https://github.com/psf/requests/issues/2155#issuecomment-50771010
                import functools
                res.raw.read = functools.partial(
                    res.raw.read,
                    decode_content=True)

                # TODO: 파일 길이가 적합한지?
                shutil.copyfileobj(res.raw, file, length=16*1024*1024)


def has_checkpoint() -> bool:
    workspace = Path(WORKSPACE)
    for p in workspace.joinpath('models', 'Stable-diffusion').glob('**/*'):
        if p.suffix != '.ckpt' and p.suffix != '.safetensors':
            continue

        # aria2 로 받다만 파일이면 무시하기
        if p.with_suffix(p.suffix + '.aria2c').exists():
            continue

        return True
    return False


def setup_webui() -> None:
    need_clone = True

    path = Path('repository')

    # 이미 디렉터리가 존재한다면 정상적인 레포인지 확인하기
    if path.is_dir():
        try:
            # 사용자 파일만 남겨두고 레포지토리 초기화하기
            # https://stackoverflow.com/a/12096327
            execute(
                'git stash && git pull',
                cwd='repository')

            need_clone = False

        except:
            log('레포지토리가 잘못됐습니다, 디렉터리를 제거합니다')

    if need_clone:
        shutil.rmtree(path, ignore_errors=True)
        execute(['git', 'clone', OPTIONS['REPO_URL'], str(path)])

    # 특정 커밋이 지정됐다면 체크아웃하기
    if OPTIONS['REPO_COMMIT'] != '':
        execute(
            ['git', 'checkout', OPTIONS['REPO_COMMIT']],
            cwd=path)

    if IN_COLAB:
        # Gradio 에서 앱이 위치한 경로와 다른 장치에 있는 내부 파일 접근시 발생하던 ValueError 를 해결하는 스크립트
        download(
            'https://raw.githubusercontent.com/toriato/easy-stable-diffusion/main/scripts/fix_gradio_route.py',
            'repository/scripts/fix_gradio_route.py',
            ignore_aria2=True)

        # 모델 변경 전 임시 폴더로 옮기는 스크립트
        download(
            'https://raw.githubusercontent.com/toriato/easy-stable-diffusion/main/scripts/alternate_load_model_weights.py',
            'repository/scripts/alternate_load_model_weights.py',
            ignore_aria2=True)


def parse_webui_output(line: str, log_index: Optional[int]) -> None:
    global NGROK_URL

    # 하위 파이썬 실행 중 오류가 발생하면 전체 기록 표시하기
    # TODO: 더 나은 오류 핸들링, 잘못된 내용으로 트리거 될 수 있음
    if LOG_WIDGET and 'Traceback (most recent call last):' in line:
        assert log_index
        LOG_BLOCKS[log_index]['max_childs'] = 0
        render_log()
        return

    # 내부에서 재시작할 때 ngrok 세션이 존재해도 다시 열려고 시도하는데
    # 사용 중인 토큰이 무료 플랜이면 한 세션만 열 수 있다며 뻑 나는 경우가 있음
    # 애초에 세션이 여러 개 생기면 안되긴 하지만... 일단 자동좌가 수정해주거나 PR 넣는 방법 밖엔 없을듯
    # NGROK_URL 변수가 비어있을 때만 오류 처리하도록 수정함
    if NGROK_URL == None and line == 'Invalid ngrok authtoken, ngrok connection aborted.\n':
        raise Exception('ngrok 인증 토큰이 잘못됐습니다, 올바른 토큰을 입력하거나 토큰 값 없이 실행해주세요')

    # 로컬 웹 서버가 열렸을 때
    if line.startswith('Running on local URL:'):
        if GRADIO_PASSWORD_GENERATED:
            # gradio 인증
            log(
                '\n'.join([
                    'Gradio 인증 비밀번호가 자동으로 생성됐습니다',
                    f"아이디: {OPTIONS['GRADIO_USERNAME']}",
                    f"비밀번호: {OPTIONS['GRADIO_PASSWORD']}"
                ]),
                LOG_WIDGET_STYLES['dialog_success'],
                print_to_file=False)

        # ngork
        if OPTIONS['NGROK_API_TOKEN'] != '':
            # 이전 로그에서 ngrok 주소가 표시되지 않았다면 ngrok 관련 오류 발생한 것으로 판단
            if NGROK_URL == None:
                raise Exception('ngrok 터널을 여는 중 알 수 없는 오류가 발생했습니다')

            if LOG_WIDGET:
                log(
                    '\n'.join([
                        '성공적으로 ngrok 터널이 열렸습니다',
                        NGROK_URL if LOG_WIDGET is None else f'<a target="_blank" href="{NGROK_URL}">{NGROK_URL}</a>',
                    ]),
                    LOG_WIDGET_STYLES['dialog_success'])
            else:
                log(f'성공적으로 ngrok 터널이 열렸습니다: {NGROK_URL}')

        return

    # 외부 주소 출력되면 성공적으로 실행한 것으로 판단
    if line.startswith('ngrok connected to') or line.startswith('Running on public URL:'):
        matches = re.search(r'https?://.+', line)
        assert matches
        url = matches[0]

        # gradio 는 웹 서버가 켜진 이후 바로 나오기 때문에 사용자에게 바로 보여줘도 상관 없음
        if 'gradio' in url:
            if LOG_WIDGET:
                log(
                    '\n'.join([
                        '성공적으로 Gradio 터널이 열렸습니다',
                        '<a target="_blank" href="https://arca.live/b/aiart/60683088">Gradio 는 느리고 버그가 있으므로 ngrok 사용을 추천합니다</a>',
                        f'<a target="_blank" href="{url}">{url}</a>',
                    ]),
                    LOG_WIDGET_STYLES['dialog_warning'])
            else:
                log(f'성공적으로 Gradio 터널이 열렸습니다: {url}')

        # ngork 는 우선 터널이 시작되고 이후에 웹 서버가 켜지기 때문에
        # 미리 주소를 저장해두고 이후에 로컬호스트 주소가 나온 뒤에 사용자에게 알려야함
        if 'ngrok.io' in url:
            NGROK_URL = url

        return


def start_webui(args: List[str] = OPTIONS['ARGS'], env: Dict[str, str] = {}) -> None:
    global GRADIO_PASSWORD_GENERATED

    workspace = Path(WORKSPACE).resolve()

    # 기본 인자 만들기
    if len(args) < 1:
        # xformers
        if OPTIONS['USE_XFORMERS'] and torch.cuda.is_available():
            args += [
                '--xformers',
                '--xformers-flash-attention'
            ]

        # gradio
        if OPTIONS['USE_GRADIO']:
            args += ['--share']

        # gradio 인증
        if OPTIONS['GRADIO_USERNAME'] != '':
            # 다계정이 아니고 비밀번호가 없다면 무작위로 만들기
            if OPTIONS['GRADIO_PASSWORD'] == '' and ';' not in OPTIONS['GRADIO_USERNAME']:
                from secrets import token_urlsafe
                OPTIONS['GRADIO_PASSWORD'] = token_urlsafe(8)
                GRADIO_PASSWORD_GENERATED = True

            args += [
                f'--gradio-auth',
                OPTIONS['GRADIO_USERNAME'] +
                ('' if OPTIONS['GRADIO_PASSWORD'] ==
                 '' else ':' + OPTIONS['GRADIO_PASSWORD'])
            ]

        # ngrok
        if OPTIONS['NGROK_API_TOKEN'] != '':
            args += [
                '--ngrok', OPTIONS['NGROK_API_TOKEN'],
                '--ngrok-region', 'jp'
            ]

        if WORKSPACE:
            args += [
                '--data-dir',
                str(workspace)
            ]

    # 추가 인자
    args += OPTIONS['EXTRA_ARGS']

    execute(
        [OPTIONS['PYTHON_EXECUTABLE'] or 'python', '-m', 'launch', *args],
        parser=parse_webui_output,
        cwd='repository',
        env={
            **os.environ,
            'PYTHONUNBUFFERED': '1',
            'HF_HOME': str(workspace.joinpath('cache', 'huggingface')),
            **env
        })


try:
    setup_environment()

    # 3단 이상(?) 레벨에서 실행하면 nested 된 asyncio 이 문제를 일으킴
    # 런타임을 종료해도 코랩 페이지에선 런타임이 실행 중(Busy)인 것으로 표시되므로 여기서 실행함
    if OPTIONS['DISCONNECT_RUNTIME']:
        hook_runtime_disconnect()

    setup_webui()
    start_webui()

# ^c 종료 무시하기
except KeyboardInterrupt:
    pass

except:
    # 로그 위젯이 없다면 평범하게 오류 처리하기
    if not LOG_WIDGET:
        raise

    log_trace()
