import os
import sys
import shutil
import tempfile
import re
import shlex
import json
import requests
import torch

from typing import Dict, Union, Callable, Tuple, List
from subprocess import Popen, PIPE, STDOUT
from distutils.spawn import find_executable
from importlib.util import find_spec
from pathlib import Path
from io import FileIO
from datetime import datetime

OPTIONS = {}


#@markdown ### <font color="orange">***작업 디렉터리 경로***</font>
#@markdown 임베딩, 모델, 결과와 설정 파일 등이 영구적으로 보관될 디렉터리 경로
영구_경로 = 'SD' #@param {type:"string"}
OPTIONS['WORKSPACE'] = 영구_경로

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
#@markdown - <font color="red">**단점**</font>: 접속이 느리고 끊키거나 버튼이 안 눌리는 등 오류 빈도가 높음
USE_GRADIO = True #@param {type:"boolean"}
OPTIONS['USE_GRADIO'] = USE_GRADIO

#@markdown ##### <font color="orange">***Gradio 인증 정보***</font>
#@markdown Gradio 접속 시 사용할 사용자 아이디와 비밀번호
#@markdown <br>`GRADIO_USERNAME` 입력 란에 `user1:pass1,user,pass2`처럼 입력하면 여러 사용자 추가 가능
#@markdown <br>`GRADIO_USERNAME` 입력 란을 <font color="red">비워두면</font> 인증 과정을 사용하지 않음
#@markdown <br>`GRADIO_PASSWORD` 입력 란을 <font color="red">비워두면</font> 자동으로 비밀번호를 생성함
GRADIO_USERNAME = '' #@param {type:"string"}
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

#@markdown ##### <font color="orange">***WebUI 레포지토리 분기***</font>
#@markdown 레포지토리 분기목록 [ [클릭](https://github.com/AUTOMATIC1111/stable-diffusion-webui/branches) ]
#@markdown - 추천
#@markdown <br><font color="green">1.</font> : master = 기본 webui 저장소
#@markdown <br><font color="green">2.</font> : disable_initialization = 모델 속도 업그레이드 저장소
REPO_URL_TREE = 'master' #@param {type:"string"}
OPTIONS['REPO_URL_TREE'] = REPO_URL_TREE

#@markdown ##### <font color="orange">***WebUI 레포지토리 커밋 해시***</font>
#@markdown 업데이트가 실시간으로 올라올 때 최신 버전에서 오류가 발생할 때 [레포지토리 커밋 목록](https://github.com/AUTOMATIC1111/stable-diffusion-webui/commits/master)에서
#@markdown <br>과거 커밋 해시 값[(영문과 숫자로된 난수 값; 예시 이미지)](https://vmm.pw/MzMy)을 아래에 붙여넣은 뒤 실행하면 과거 버전을 사용할 수 있음
#@markdown <br>입력 란을 <font color="red">비워두면</font> 가장 최신 커밋을 가져옴
REPO_COMMIT = '' #@param {type:"string"}
OPTIONS['REPO_COMMIT'] = REPO_COMMIT

#@markdown ##### <font color="orange">***WebUI 추가 인자***</font>
#@markdown [사용할 수 있는 인자 목록](https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/shared.py#L23)
EXTRA_ARGS = '' #@param {type:"string"}
OPTIONS['EXTRA_ARGS'] = shlex.split(EXTRA_ARGS)

#@markdown ##### <font color="orange">***모델 및 VAE 다운 설정***</font>

DOWNLOADS = True #@param {type:"boolean"}


#@markdown ##### <font color="orange">***모델다운 링크***</font>
#@markdown 추천모델 다운 허깅 링크 [ [ 링크 ](https://huggingface.co/Kaeya/aichan_blend/tree/main) ]
MODEl_URL = 'https://huggingface.co/Kaeya/aichan_blend/resolve/main/Anything3.0%2BF222-SD1.4-pruned.safetensors' #@param {type:"string"}
OPTIONS['MODEl_URL'] = MODEl_URL
MODEl_NAME = 'Anything3.0%2BF222-SD1.4-pruned.safetensors' #@param {type:"string"}
OPTIONS['MODEl_NAME'] = MODEl_NAME

#@markdown ##### <font color="orange">***VAE 다운링크***</font>
#@markdown 추천VAE 다운 허깅 링크 [ [ 링크 ](https://huggingface.co/Kaeya/aichan_blend/tree/main/vae) ]
VAE_URL = 'https://huggingface.co/Kaeya/aichan_blend/resolve/main/vae/Anything-V3.0.vae.safetensors' #@param {type:"string"}
OPTIONS['VAE_URL'] = VAE_URL
VAE_NAME = 'Anything-V3.0.vae.safetensors' #@param {type:"string"}
OPTIONS['VAE_NAME'] = VAE_NAME

#####################################################
# 사용자 설정 값 끝
#####################################################

# 작업 디렉터리 <-> 레포지토리 심볼릭 중 제외할 경로
SYMLINK_BLACKLIST = (
    # 구동에 불필요하면 파일 및 디렉터리
    '.',
    'cache',
    'logs',
    'override.json',

    # 레포지토리 자체
    'repository',

    # 크거나 많은 파일이 담겨져 있기 때문에 수동으로 만듦
    'extensions',
    'models',
    'outputs'
)

# 임시 디렉터리
TEMP_DIR = tempfile.mkdtemp()

# 로그 파일
LOG_FILE: FileIO = None

# 로그 HTML 위젯
LOG_WIDGET = None

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

def setup_colab():
    # 터널링 서비스가 아예 존재하지 않다면 오류 반환하기
    assert OPTIONS['USE_GRADIO'] or OPTIONS['NGROK_API_TOKEN'] != '', '터널링 서비스를 하나 이상 선택해주세요'

    # 코랩 환경인데 글카가 왜 없어...?
    assert torch.cuda.is_available(), 'GPU 가 없습니다, 런타임 유형이 잘못됐거나 GPU 할당량이 초과된 것 같습니다'

    # 빠른 다운로드를 위해 aria2 패키지 설치
    if not find_executable('aria2c'):
        execute(
            ['apt', 'install', 'aria2'],
            summary='빠른 다운로드를 위해 aria2 패키지를 설치합니다',
            throw=False
        )

    OPTIONS['EXTRA_ARGS'] += [
        # 메모리가 낮아 모델을 VRAM 위로 올려 사용해야함
        '--lowram',

        # --listen 또는 --share 인자를 사용하면 확장 기능 탭이 막혀버림
        # 어처피 Gradio 비밀번호는 자동으로 생성되는데
        # 일부러 제거하고 외부 접근 공개한 바보 책임이니 인자 넣어둠
        '--enable-insecure-extension-access'
    ]

    # 코랩 환경에서 이유는 알 수 없지만 /usr 디렉터리 내에서 읽기/쓰기 속도가 다른 곳보다 월등히 빠름
    # 아마 /content 에 큰 용량을 박아두는 사용하는 사람들이 많아서 그런듯...?
    src = Path('/usr/local/repository')
    dst = Path('repository').absolute()
    delete(dst)
    dst.symlink_to(src, True)

    # huggingface 모델 캐시 심볼릭 만들기
    src = Path('cache', 'huggingface').absolute()
    dst = Path('/root/.cache/huggingface')
    delete(dst)
    src.mkdir(0o777, True, True)
    dst.symlink_to(src, True)



def setup_environment():
    global LOG_WIDGET

    # 노트북 환경이라면 로그 표시를 위한 HTML 요소 만들기
    if 'ipykernel' in sys.modules:
        try:
            from IPython.display import display
            from ipywidgets import widgets

            LOG_WIDGET = widgets.HTML()
            LOG_WIDGET.blocks = []

            display(LOG_WIDGET)
        except:
            pass

    # google.colab 패키지가 있다면 코랩 환경으로 인식하기
    if has_python_package('google') and has_python_package('google.colab'):
        # 코랩 노트북의 경우 작업 디렉터리로 항상 /content 디렉터리를 사용함
        cwd = Path('/content')

        # 구글 드라이브 마운트하기
        if OPTIONS['USE_GOOGLE_DRIVE']:
            from google.colab import drive
            drive.mount(str(cwd.joinpath('drive')))

            # 마운트한 디렉터리 속 MyDrive 디렉터리부터 쓰기 가능함
            cwd = cwd.joinpath('drive', 'MyDrive')

        chdir(cwd.joinpath(OPTIONS['WORKSPACE']))
        setup_colab()
    else:
        chdir(OPTIONS['WORKSPACE'])

    # 현재 환경 출력
    import platform
    log(platform.platform())
    log(f'Python {platform.python_version()}')
    log(str(Path.cwd()))

    # 체크포인트 모델이 존재하지 않는다면 기본 모델 받아오기
    if not has_checkpoint():
        if DOWNLOADS:
            for file in [
                {
                    'url': OPTIONS['MODEl_URL'],
                    'target': f'models/Stable-diffusion/{OPTIONS["MODEl_NAME"]}',
                    'summary': '체크포인트 모델 파일을 받아옵니다'
                },
                {
                    'url': OPTIONS['VAE_URL'],
                    'target': f'models/VAE/{OPTIONS["VAE_NAME"]}',
                    'summary': 'VAE 모델 파일을 받아옵니다'
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
    html = f'''<div style="{format_styles(LOG_WIDGET_STYLES['wrapper'])}">'''

    for block in LOG_WIDGET.blocks:
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
    parent_index: int = None,
    child_styles={
        'padding-left': '1em',
        'color': 'gray'
    },
    max_childs=0,

    print_to_file=True,
    print_to_widget=True
) -> Tuple[None, int]:

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
            LOG_WIDGET.blocks.append({
                'msg': msg,
                'styles': styles,
                'childs': [],
                'child_styles': child_styles,
                'max_childs': max_childs
            })
            render_log()
            return len(LOG_WIDGET.blocks) - 1

        # 부모 로그가 존재한다면 추가하기
        LOG_WIDGET.blocks[parent_index]['childs'].append(msg)
        render_log()
        return

    print('\t' if parent_index else '' + msg, end='')
    return


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

    parent_index = log('보고서를 만들고 있습니다...', styles)

    # 오류가 존재한다면 오류 정보와 스택 트레이스 출력하기
    if ex_type is not None:
        log(parent_index=parent_index, msg=f'{ex_type.__name__}: {ex_value}')
        log(
            parent_index=parent_index,
            msg=format_list(
                map(
                    lambda v: f'{v[0]}#{v[1]}\n\t{v[2]}\n\t{v[3]}',
                    traceback.extract_tb(ex_traceback)
                )
            )
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
        res = requests.post('https://hastebin.com/documents',
                            data=logs.encode('utf-8'))
        url = f"https://hastebin.com/raw/{json.loads(res.text)['key']}"

        # 기존 오류 메세지 업데이트
        LOG_WIDGET.blocks[parent_index]['msg'] = '\n'.join([
            '오류가 발생했습니다, 아래 주소를 <a href="https://discord.gg/6wQeA2QXgM">디스코드 서버</a>에 보고해주세요',
            f'<a target="_blank" href="{url}">{url}</a>',
        ])

        render_log()


# ==============================
# 서브 프로세스
# ==============================
running_subprocess = None


def execute(
    args: Union[str, List[str]],
    parser: Callable = None,
    summary: str = None,
    hide_summary=False,
    print_to_file=True,
    print_to_widget=True,
    throw=True,
    **kwargs
) -> Tuple[str, Popen]:

    global running_subprocess

    # 이미 서브 프로세스가 존재한다면 예외 처리하기
    if running_subprocess and running_subprocess.poll() is None:
        raise Exception('이미 다른 하위 프로세스가 실행되고 있습니다')

    # 서브 프로세스 만들기
    running_subprocess = Popen(
        args,
        stdout=PIPE,
        stderr=STDOUT,
        encoding='utf-8',
        **kwargs,
    )
    running_subprocess.output = ''

    # 로그에 시작한 프로세스 정보 출력하기
    formatted_args = args if isinstance(args, str) else ' '.join(args)
    summary = formatted_args if summary is None else f'{summary}\n   {formatted_args}'

    running_subprocess.parent_index = log(
        f'=> {summary}',
        styles={'color': 'yellow'},
        max_childs=5,
    )

    # 프로세스 출력 위젯에 리다이렉션하기
    while running_subprocess.poll() is None:
        # 출력이 비어있다면 넘어가기
        line = running_subprocess.stdout.readline()
        if not line:
            continue

        # 프로세스 출력 버퍼에 추가하기
        running_subprocess.output += line

        # 파서 함수 실행하기
        if callable(parser):
            try:
                if parser(line):
                    continue
            except:
                log_trace()

        # 프로세스 출력 로그하기
        log(
            line,
            newline=False,
            parent_index=running_subprocess.parent_index,
            print_to_file=print_to_file,
            print_to_widget=print_to_widget
        )

    # 변수 정리하기
    output = running_subprocess.output
    returncode = running_subprocess.poll()

    # 로그 블록 업데이트
    if LOG_WIDGET:
        if returncode == 0:
            # 성공적으로 프로세스가 종료됐을 때
            if hide_summary:
                # 현재 로그 블록 숨기기 (제거하기)
                del LOG_WIDGET.blocks[running_subprocess.parent_index]
            else:
                # 현재 로그 텍스트 초록색으로 변경하고 프로세스 출력 숨기기
                LOG_WIDGET.blocks[running_subprocess.parent_index]['styles']['color'] = 'green'
                LOG_WIDGET.blocks[running_subprocess.parent_index]['max_childs'] = None
        else:
            # 현재 로그 텍스트 빨간색으로 변경하고 프로세스 출력 모두 표시하기
            LOG_WIDGET.blocks[running_subprocess.parent_index]['styles']['color'] = 'red'
            LOG_WIDGET.blocks[running_subprocess.parent_index]['max_childs'] = 0

        # 로그 블록 렌더링
        render_log()

    # 오류 코드를 반환했다면
    if returncode != 0 and throw:
        raise Exception(f'프로세스가 {returncode} 코드를 반환했습니다')

    return output, returncode

# ==============================
# 작업 경로
# ==============================


def chdir(cwd: Path) -> None:
    global LOG_FILE

    cwd = cwd.absolute()

    # 작업 경로 변경
    old_cwd = Path.cwd().absolute()
    cwd.mkdir(0o777, True, True)
    os.chdir(cwd)

    # 기존 로그 파일 옮기기
    os.makedirs('logs', exist_ok=True)

    log_path = Path('logs', datetime.strftime(
        datetime.now(), '%Y-%m-%d_%H-%M-%S.log'))

    if LOG_FILE:
        LOG_FILE.close()
        Path(old_cwd, LOG_FILE.name).rename(log_path)

    LOG_FILE = log_path.open('a')

    # 덮어쓸 설정 파일 가져오기
    override_path = cwd.joinpath('override.json')
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


def delete(path: os.PathLike) -> None:
    path = Path(path)

    if path.is_file() or path.is_symlink():
        path.unlink()
    else:
        shutil.rmtree(path, ignore_errors=True)


def has_python_package(pkg: str, check_loader=True) -> bool:
    spec = find_spec(pkg)
    return spec and (check_loader and spec.loader is not None)


# ==============================
# 파일 다운로드
# ==============================
def download(url: str, target: str, **kwargs):
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
            **kwargs
        )

    elif find_executable('curl'):
        execute(
            [
                'curl',
                '--location',
                '--output', target,
                url
            ],
            **kwargs
        )

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
                    decode_content=True
                )

                # TODO: 파일 길이가 적합한지?
                shutil.copyfileobj(res.raw, file, length=16*1024*1024)


def has_checkpoint() -> bool:
    for p in Path('models', 'Stable-diffusion').glob('**/*'):
        if p.suffix != '.ckpt' and p.suffix != '.safetensors':
            continue

        # aria2 로 받다만 파일이면 무시하기
        if p.with_suffix(p.suffix + '.aria2c').exists():
            continue

        return True
    return False


# ==============================
# WebUI 레포지토리 및 종속 패키지 설치
# ==============================
def patch_webui_repository() -> None:

    # 기본 파일 만들기
    for path, content in {
        'config.json': json.dumps({
            'CLIP_stop_at_last_layers': 2
        }),
        'ui-config.json': json.dumps({
            'txt2img/Prompt/value': 'best quality, masterpiece',
            'txt2img/Negative prompt/value': 'lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry',
            'txt2img/Sampling Steps/value': 28,
            'txt2img/Width/value': 512,
            'txt2img/Height/value': 768,
            'txt2img/CFG Scale/value': 12
        }),
        'styles.csv': ''
    }.items():
        path = Path(path)
        if not path.exists():
            path.write_text(content)

    # Gradio 에서 앱이 위치한 경로와 다른 장치에 있는 내부 파일 접근시 발생하던 ValueError 를 해결하는 스크립트
    download(
        'https://raw.githubusercontent.com/toriato/easy-stable-diffusion/main/scripts/fix_gradio_route.py',
        'repository/scripts/fix_gradio_route.py',
        summary='Gradio 경로와 관련된 버그 픽스 스크립트를 받아옵니다'
    )

    # 고정 심볼릭 링크 만들기
    for src in ['extensions', 'models', 'outputs']:
        src = Path(src)
        dst = Path('repository', src)

        # 목표가 심볼릭 링크라면 실제 주소 가져오기
        if src.is_symlink():
            src = os.readlink(src)

        if not src.exists():
            src.mkdir(0o777, True, True)

        # 기존 파일/심볼릭 링크 또는 디렉터리 제거하기
        delete(dst)

        dst.symlink_to(src.absolute(), src.is_dir())

    # 가변 심볼릭 링크 만들기
    for root, dirs, files in os.walk('.'):
        # 디렉터리 필터링하기
        for dir in dirs[:]:
            src = Path(root, dir)
            dst = Path('repository', src)

            # 디렉터리가 블랙리스트에 포함됐는지 확인하기
            for blacklist in SYMLINK_BLACKLIST:
                if str(src).startswith(blacklist):
                    break

            else:
                # 목표 디렉터리가 존재한다면 하위에서 만드므로 넘어가기
                if not dst.is_symlink() and dst.is_dir():
                    continue

                # 이미 존재하면 심볼릭 링크를 만들 수 없으므로 기존 파일 제거하기
                delete(dst)

                dst.symlink_to(src.absolute(), True)

            # os.walk 에서 처리하지 않도록 삭제
            # TODO: List.remove = 존나 느림, 파이썬 병신~ (https://stackoverflow.com/a/34238688)
            dirs.remove(dir)

        for src in files:
            src = Path(root, src)
            dst = Path('repository', src)

            # 파일이 블랙리스트에 포함됐는지 확인하기
            for blacklist in SYMLINK_BLACKLIST:
                if str(src).startswith(blacklist):
                    break

            else:
                # 목표가 심볼릭 링크라면 실제 주소 가져오기
                if src.is_symlink():
                    src = os.readlink(src)

                    # 심볼릭 링크가 잘못된 경로를 가르키고 있었다면 무시하기
                    if not src.exists():
                        continue

                # 기존 파일/심볼릭 링크 또는 디렉터리 제거하기
                delete(dst)

                # 심볼릭 링크 만들기
                dst.symlink_to(src.absolute())


def setup_webui() -> None:
    need_clone = True

    path = Path('repository')

    # 이미 디렉터리가 존재한다면 정상적인 레포인지 확인하기
    if path.is_dir():
        try:
            # 사용자 파일만 남겨두고 레포지토리 초기화하기
            # https://stackoverflow.com/a/12096327
            execute(
                'git reset --hard HEAD && git pull',
                summary='레포지토리를 업데이트 합니다',
                shell=True,
                cwd='repository'
            )

            need_clone = False

        except:
            log('레포지토리가 잘못됐습니다, 디렉터리를 제거합니다')

    if need_clone:
        # 실제 레포지토리 경로 가져오기
        # 코랩 환경에선 레포지토리 경로를 심볼릭 링크하기 때문에 경로를 가져와야함
        if path.is_symlink():
            path = os.readlink(path)

        shutil.rmtree(path, ignore_errors=True)
        execute(
            ['git', 'clone','-b', OPTIONS['REPO_URL_TREE'], '--single-branch', OPTIONS['REPO_URL'], str(path)],
            summary='레포지토리를 가져옵니다'
        )
    # 특정 커밋이 지정됐다면 체크아웃하기
    if OPTIONS['REPO_COMMIT'] != '':
        execute(
            ['git', 'checkout', OPTIONS['REPO_COMMIT']],
            summary=f"레포지토리를 {OPTIONS['REPO_COMMIT']} 커밋으로 되돌립니다",
            cwd=path
        )

    patch_webui_repository()


def parse_webui_output(line: str) -> bool:
    global NGROK_URL

    # 하위 파이썬 실행 중 오류가 발생하면 전체 기록 표시하기
    # TODO: 더 나은 오류 핸들링, 잘못된 내용으로 트리거 될 수 있음
    if LOG_WIDGET and 'Traceback (most recent call last):' in line:
        LOG_WIDGET.blocks[running_subprocess.parent_index]['max_childs'] = 0
        render_log()
        return

    if line == 'paramiko.ssh_exception.SSHException: Error reading SSH protocol banner[Errno 104] Connection reset by peer\n':
        raise Exception('Gradio 연결 중 알 수 없는 오류가 발생했습니다, 다시 실행해주세요')

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
                print_to_file=False
            )

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
                    LOG_WIDGET_STYLES['dialog_success']
                )
            else:
                log(f'성공적으로 ngrok 터널이 열렸습니다: {NGROK_URL}')

        return

    # 외부 주소 출력되면 성공적으로 실행한 것으로 판단
    if line.startswith('ngrok connected to') or line.startswith('Running on public URL:'):
        url = re.search(r'https?://.+', line)[0]

        # gradio 는 웹 서버가 켜진 이후 바로 나오기 때문에 사용자에게 바로 보여줘도 상관 없음
        if 'gradio' in url:
            if LOG_WIDGET:
                log(
                    '\n'.join([
                        '성공적으로 Gradio 터널이 열렸습니다',
                        '<a target="_blank" href="https://arca.live/b/aiart/60683088">Gradio 는 느리고 버그가 있으므로 ngrok 사용을 추천합니다</a>',
                        f'<a target="_blank" href="{url}">{url}</a>',
                    ]),
                    LOG_WIDGET_STYLES['dialog_warning']
                )
            else:
                log(f'성공적으로 Gradio 터널이 열렸습니다: {url}')

        # ngork 는 우선 터널이 시작되고 이후에 웹 서버가 켜지기 때문에
        # 미리 주소를 저장해두고 이후에 로컬호스트 주소가 나온 뒤에 사용자에게 알려야함
        if 'ngrok.io' in url:
            NGROK_URL = url

        return


def start_webui(args: List[str] = None, env: Dict[str, str] = None) -> None:
    global GRADIO_PASSWORD_GENERATED

    # 기본 환경 변수 만들기
    if env is None:
        env = {
            **os.environ,
            'PYTHONUNBUFFERED': '1',
            'REQS_FILE': 'requirements.txt',
        }

    # 기본 인자 만들기
    if args is None:
        # launch.py 실행할 땐 레포지토리 경로에서 실행해야하기 때문에
        # 현재 작업 디렉터리를 절대 경로로 가져와 인자로 보내줄 필요가 있음
        args = [
            # TODO: 기븐으로 설정 해둬도 괜찮을까...?
            '--gradio-img2img-tool', 'color-sketch',
        ]

        # xformers
        if OPTIONS['USE_XFORMERS']:
            if not has_python_package('xformers'):
                execute(
                    [
                        'pip', 'install',
                        'https://github.com/toriato/easy-stable-diffusion/releases/download/xformers-e163309/xformers-0.0.15+e163309.cu116.d20221226-cp38-cp38-linux_x86_64.whl'
                    ],
                    summary='xformers 패키지를 설치합니다',
                    throw=False
                )

            if has_python_package('xformers'):
                args.append('--xformers')

        # gradio
        if OPTIONS['USE_GRADIO']:
            args.append('--share')

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

        # 추가 인자
        args += OPTIONS['EXTRA_ARGS']

    execute(
        ['python', 'launch.py', *args],
        parser=parse_webui_output,
        cwd='repository',
        env=env
    )


try:
    setup_environment()
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
