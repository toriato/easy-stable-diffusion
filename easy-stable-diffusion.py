# @title
# fmt: off
import os
import sys
import shutil
import platform
import tempfile
import re
import json
import requests
from typing import Dict, Union, Callable, Tuple, List
from subprocess import Popen, PIPE, STDOUT
from distutils.spawn import find_executable
from importlib.util import find_spec
from pathlib import Path
from io import FileIO
from datetime import datetime

OPTIONS = {}

# @markdown ### <font color="orange">***작업 디렉터리 경로***</font>
# @markdown 임베딩, 모델, 결과와 설정 파일 등이 영구적으로 보관될 디렉터리 경로
WORKSPACE = 'SD' # @param {type:"string"}
OPTIONS['WORKSPACE'] = WORKSPACE

# @markdown ##### <font size="2" color="red">(선택)</font> <font color="orange">***구글 드라이브와 동기화할지?***</font>
# @markdown <font color="red">**주의**</font>: 동기화 전 남은 용량이 충분한지 확인 필수 (5GB 이상)
USE_GOOGLE_DRIVE = True  # @param {type:"boolean"}
OPTIONS['USE_GOOGLE_DRIVE'] = USE_GOOGLE_DRIVE

# @markdown ##### <font size="2" color="red">(선택)</font> <font color="orange">***xformers 를 사용할지?***</font>
# @markdown - <font color="green">장점</font>: 성능 10% 향상
# @markdown - <font color="red">단점</font>: 미리 빌드한 패키지가 지원하지 않는 환경에선 빌드해 사용해야함 (매우 느림)
# @markdown - <font color="red">단점</font>: 출력한 그림의 질이 조금 떨어질 수 있음
USE_XFORMERS = True  # @param {type:"boolean"}
OPTIONS['USE_XFORMERS'] = USE_XFORMERS

# @markdown ##### <font size="2" color="red">(선택)</font> <font color="orange">***deepbooru 를 사용할지?***</font>
# @markdown IMG2IMG 이미지를 프롬프트로 사용할 있게 단부루 태그로 변환(예측)하는 기능
# @markdown - <font color="red">단점</font>: 처음 실행할 때 추가 패키지를 받기 때문에 시간이 조금 더 걸림
USE_DEEPBOORU = True  # @param {type:"boolean"}
OPTIONS['USE_DEEPBOORU'] = USE_DEEPBOORU

# @markdown ##### <font size="2" color="red">(선택)</font> <font color="orange">***사용자 스크립트를 받아올지?***</font>
# @markdown 실행할 때 공식 레포지토리 위키에 [소개된 사용자 스크립트](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Custom-Scripts)을 받아옴
# @markdown <br>`SD/scripts` 경로에 원하는 스크립트를 넣으면 덮어쓸 수 있음
DOWNLOAD_CUSTOM_SCRIPTS = True  # @param {type:"boolean"}
OPTIONS['DOWNLOAD_CUSTOM_SCRIPTS'] = DOWNLOAD_CUSTOM_SCRIPTS

# @markdown ##### <font size="2" color="red">(선택)</font> <font color="orange">***확장 기능을 받아올지?***</font>
# @markdown 실행할 때 공식 레포지토리 위키에 [소개된 확장 기능](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Extensions)을 받아옴
# @markdown <br>`SD/extensions` 경로에 원하는 확장 기능을 넣어두면 덮어쓸 수 있음
DOWNLOAD_EXTENSIONS = True  # @param {type:"boolean"}
OPTIONS['DOWNLOAD_EXTENSIONS'] = DOWNLOAD_EXTENSIONS

# @markdown ##### <font size="2" color="red">(선택)</font> <font color="orange">***Gradio 터널을 사용할지?***</font>
# @markdown - <font color="green">장점</font>: 따로 설정할 필요가 없어 편리함
# @markdown - <font color="red">**단점**</font>: 접속이 느리고 끊키거나 버튼이 안 눌리는 등 오류 빈도가 높음
USE_GRADIO = True # @param {type:"boolean"}
OPTIONS['USE_GRADIO'] = USE_GRADIO

# @markdown ##### <font size="2" color="red">(선택)</font> <font color="orange">***Gradio 인증 정보***</font>
# @markdown Gradio 접속 시 사용할 사용자 아이디와 비밀번호
# @markdown <br>`GRADIO_USERNAME` 입력 란에 `user1:pass1,user,pass2`처럼 입력하면 여러 사용자 추가 가능
# @markdown <br>`GRADIO_USERNAME` 입력 란을 <font color="red">비워두면</font> 인증 과정을 사용하지 않음
# @markdown <br>`GRADIO_PASSWORD` 입력 란을 <font color="red">비워두면</font> 자동으로 비밀번호를 생성함
GRADIO_USERNAME = 'gradio' # @param {type:"string"}
GRADIO_PASSWORD = '' # @param {type:"string"}
GRADIO_PASSWORD_GENERATED = False
OPTIONS['GRADIO_USERNAME'] = GRADIO_USERNAME
OPTIONS['GRADIO_PASSWORD'] = GRADIO_PASSWORD

# @markdown ##### <font size="2" color="red">(선택)</font> <font color="orange">***ngrok API 키***</font>
# @markdown ngrok 터널에 사용할 API 토큰
# @markdown <br>[설정하는 방법은 여기를 클릭해 확인](https://arca.live/b/aiart/60683088), [API 토큰은 여기를 눌러 계정을 만든 뒤 얻을 수 있음](https://dashboard.ngrok.com/get-started/your-authtoken)
# @markdown <br>입력 란을 <font color="red">비워두면</font> ngrok 터널을 비활성화함
# @markdown - <font color="green">장점</font>: 접속이 빠른 편이고 타임아웃이 거의 발생하지 않음
# @markdown - <font color="red">**단점**</font>: 계정을 만들고 API 토큰을 직접 입력해줘야함
NGROK_API_TOKEN = '' # @param {type:"string"}
NGROK_URL = None
OPTIONS['NGROK_API_TOKEN'] = NGROK_API_TOKEN

# @markdown ##### <font size="2" color="red">(선택)</font> <font color="orange">***WebUI 레포지토리 주소***</font>
REPO_URL = 'https://github.com/AUTOMATIC1111/stable-diffusion-webui.git' # @param {type:"string"}
OPTIONS['REPO_URL'] = REPO_URL

# @markdown ##### <font size="2" color="red">(선택)</font> <font color="orange">***WebUI 레포지토리 커밋 해시***</font>
# @markdown 업데이트가 실시간으로 올라올 때 최신 버전에서 오류가 발생할 때 [레포지토리 커밋 목록](https://github.com/AUTOMATIC1111/stable-diffusion-webui/commits/master)에서
# @markdown <br>과거 커밋 해시 값[(영문과 숫자로된 난수 값; 예시 이미지)](https://vmm.pw/MzMy)을 아래에 붙여넣은 뒤 실행하면 과거 버전을 사용할 수 있음
# @markdown <br>입력 란을 <font color="red">비워두면</font> 가장 최신 커밋을 가져옴
REPO_COMMIT = '' # @param {type:"string"}
OPTIONS['REPO_COMMIT'] = REPO_COMMIT

# @markdown ##### <font size="2" color="red">(선택)</font> <font color="orange">***WebUI 추가 인자***</font>
# @markdown [사용할 수 있는 인자 목록](https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/shared.py#L23)
EXTRA_ARGS = '' # @param {type:"string"}
OPTIONS['EXTRA_ARGS'] = EXTRA_ARGS

# 받을 수 있는 체크포인트들
DEFAULT_CHECKPOINT_URLS = [
    {
        'url': 'https://pub-2fdef7a2969f43289c42ac5ae3412fd4.r2.dev/animefull-final-pruned.ckpt'
    },
    {
        'url': 'https://pub-2fdef7a2969f43289c42ac5ae3412fd4.r2.dev/animevae.pt'
    },
    {
        'url': 'https://gist.githubusercontent.com/toriato/ae1f587f4d1e9ee5d0e910c627277930/raw/6019f8782875497f6e5b3e537e30a75df5b64812/animefull-final-pruned.yaml'
    }
]

# 추가로 받을 스크립트들
PREDEFINED_SCRIPTS = [
    # Advanced prompt matrix
    # https://github.com/GRMrGecko/stable-diffusion-webui-automatic/blob/advanced_matrix/scripts/advanced_prompt_matrix.py
    lambda: download(
        'https://raw.githubusercontent.com/GRMrGecko/stable-diffusion-webui-automatic/advanced_matrix/scripts/advanced_prompt_matrix.py',
        'repository/scripts'
    ),

    # txt2mask
    # https://github.com/ThereforeGames/txt2mask
    [
        lambda: shutil.rmtree(TEMP_DIR, ignore_errors=True),
        lambda: execute(
            ['git', 'clone', 'https://github.com/ThereforeGames/txt2mask.git', TEMP_DIR],
            hide_summary=True
        ),
        lambda: shutil.rmtree('repository/repositories/clipseg', ignore_errors=True),
        lambda: shutil.copytree(os.path.join(TEMP_DIR, 'repositories/clipseg'), 'repository/repositories/clipseg'),
        lambda: shutil.copy(os.path.join(TEMP_DIR, 'scripts/txt2mask.py'), 'repository/scripts'),
        lambda: shutil.rmtree(TEMP_DIR, ignore_errors=True),
    ],

    # Img2img Video
    # https://github.com/memes-forever/Stable-diffusion-webui-video
    lambda: download(
        'https://raw.githubusercontent.com/memes-forever/Stable-diffusion-webui-video/main/videos.py',
        'repository/scripts'
    ),

    # Seed Travel
    # https://github.com/yownas/seed_travel
    [
        lambda: None if has_python_package('moviepy') else execute(['pip', 'install', 'moviepy']),
        lambda: download(
            'https://raw.githubusercontent.com/yownas/seed_travel/main/scripts/seed_travel.py',
            'repository/scripts',
        )
    ],

    # Animator
    # https://github.com/Animator-Anon/Animator
    lambda: download(
        'https://raw.githubusercontent.com/Animator-Anon/Animator/main/animation.py',
        'repository/scripts'
    ),

    # Alternate Noise Schedules
    # https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Custom-Scripts#alternate-noise-schedules
    lambda: download(
        'https://gist.githubusercontent.com/dfaker/f88aa62e3a14b559fe4e5f6b345db664/raw/791dabfa0ab26399aa2635bcbc1cf6267aa4ffc2/alternate_sampler_noise_schedules.py',
        'repository/scripts'
    ),

    # Vid2Vid
    # https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Custom-Scripts#vid2vid
    lambda: download(
        'https://raw.githubusercontent.com/Filarius/stable-diffusion-webui/master/scripts/vid2vid.py',
        'repository/scripts'
    ),

    # Shift Attention
    # https://github.com/yownas/shift-attention
    [
        lambda: None if has_python_package('moviepy') else execute(['pip', 'install', 'moviepy']),
        lambda: download(
            'https://raw.githubusercontent.com/yownas/shift-attention/main/scripts/shift_attention.py',
            'repository/scripts'
        )
    ],

    # Loopback and Superimpose
    # https://github.com/DiceOwl/StableDiffusionStuff
    lambda: download(
        'https://raw.githubusercontent.com/DiceOwl/StableDiffusionStuff/main/loopback_superimpose.py',
        'repository/scripts'
    ),

    # Run n times
    # https://gist.github.com/camenduru/9ec5f8141db9902e375967e93250860f
    lambda: download(
        'https://gist.githubusercontent.com/camenduru/9ec5f8141db9902e375967e93250860f/raw/b5c741676c5514105b9a1ea7dd438ca83802f16f/run_n_times.py',
        'repository/scripts'
    ),

    # Advanced Loopback
    # https://github.com/Extraltodeus/advanced-loopback-for-sd-webui
    lambda: download(
        'https://raw.githubusercontent.com/Extraltodeus/advanced-loopback-for-sd-webui/main/advanced_loopback.py',
        'repository/scripts'
    ),

    # prompt-morph
    # https://github.com/feffy380/prompt-morph
    [
        lambda: None if has_python_package('moviepy') else execute(['pip', 'install', 'moviepy']),
        lambda: download(
            'https://raw.githubusercontent.com/feffy380/prompt-morph/master/prompt_morph.py',
            'repository/scripts'
        ),
    ],

    # prompt interpolation
    # https://github.com/EugeoSynthesisThirtyTwo/prompt-interpolation-script-for-sd-webui
    lambda: download(
        'https://raw.githubusercontent.com/EugeoSynthesisThirtyTwo/prompt-interpolation-script-for-sd-webui/main/prompt_interpolation.py',
        'repository/scripts'
    ),

    # Asymmetric Tiling
    # https://github.com/tjm35/asymmetric-tiling-sd-webui/
    lambda: download(
        'https://raw.githubusercontent.com/tjm35/asymmetric-tiling-sd-webui/main/asymmetric_tiling.py',
        'repository/scripts'
    ),

    # Embedding to PNG
    # https://github.com/dfaker/embedding-to-png-script
    lambda: download(
        'https://raw.githubusercontent.com/dfaker/embedding-to-png-script/main/embedding_to_png.py',
        'repository/scripts'
    ),
]

# 추가로 받을 확장 기능
PREDEFINED_EXTENSIONS = [
    'https://github.com/AUTOMATIC1111/stable-diffusion-webui-aesthetic-gradients.git',
    # 'https://github.com/AUTOMATIC1111/stable-diffusion-webui-wildcards.git',
    'https://github.com/yfszzx/stable-diffusion-webui-images-browser.git',
    'https://github.com/yfszzx/stable-diffusion-webui-inspiration.git',
    'https://github.com/tsngo/stable-diffusion-webui-aesthetic-image-scorer',
    'https://github.com/Klokinator/UnivAICharGen.git',
]

# 임시 디렉터리
TEMP_DIR = tempfile.mkdtemp()

# 로그 파일
LOG_FILE: FileIO = None

# 로그 HTML 위젯
LOG_WIDGET = None

# 로그 HTML 위젯 스타일
LOG_WIDGET_STYLES = {
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

# 현재 코랩 환경에서 구동 중인지?
IN_COLAB = False

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
    styles = {
        'overflow-x': 'auto',
        'max-width': '700px',
        'padding': '1em',
        'background-color': 'black',
        'white-space': 'pre',
        'font-family': 'monospace',
        'font-size': '1em',
        'line-height': '1.1em',
        'color': 'white'
    }

    html = f'<div style="{format_styles(styles)}">'

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

def log(msg: str, styles={}, newline=True,
        parent=False, parent_index: int=None, child_styles={}, max_childs=0,
        print_to_file=True, print_to_widget=True) -> Tuple[None, int]:
    # 기록할 내용이 ngrok API 키와 일치한다면 숨기기
    # TODO: 더 나은 문자열 검사, 원치 않은 내용이 가려질 수도 있음
    if OPTIONS['NGROK_API_TOKEN'] != '':
        msg = msg.replace(OPTIONS['NGROK_API_TOKEN'], '**REDACTED**')

    if newline:
        msg += '\n'

    # 파일에 기록하기
    if print_to_file:
        if LOG_FILE:
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

        # 로그 위젯이 존재하지 않는다면 그냥 출력하기
        if not LOG_WIDGET:
            print('\t' + msg, end='')
            return

        # 부모 로그가 존재한다면 추가하기
        LOG_WIDGET.blocks[parent_index]['childs'].append(msg)
        render_log()
        return

    print(msg, end='')
    return

def log_trace() -> None:
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
        res = requests.post('https://hastebin.com/documents', data=logs.encode('utf-8'))
        url = f"https://hastebin.com/{json.loads(res.text)['key']}"

        # 기존 오류 메세지 업데이트
        LOG_WIDGET.blocks[parent_index]['msg'] = '\n'.join([
            '오류가 발생했습니다, 아래 주소를 복사해 보고해주세요',
            f'<a target="_blank" href="{url}">{url}</a>',
        ])

        render_log()


# ==============================
# 서브 프로세스
# ==============================
running_subprocess = None

def execute(args: Union[str, List[str]], parser: Callable=None,
            summary: str=None, hide_summary=False, print_to_file=True, print_to_widget=True,
            throw=True, **kwargs) -> Tuple[str, Popen]:
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
        styles={ 'color': 'yellow' },
        max_childs = 5,
    )

    # 프로세스 출력 위젯에 리다이렉션하기
    while running_subprocess.poll() is None:
        # 출력이 비어있다면 넘어가기
        line = running_subprocess.stdout.readline()
        if not line: continue

        # 프로세스 출력 버퍼에 추가하기
        running_subprocess.output += line

        # 파서 처리
        if callable(parser):
            try:
                if parser(line): continue
            except:
                log_trace()

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
            if hide_summary:
                del LOG_WIDGET.blocks[running_subprocess.parent_index]
            else:
                LOG_WIDGET.blocks[running_subprocess.parent_index]['styles']['color'] = 'green'
                LOG_WIDGET.blocks[running_subprocess.parent_index]['max_childs'] = None
        else:
            LOG_WIDGET.blocks[running_subprocess.parent_index]['styles']['color'] = 'red'
            LOG_WIDGET.blocks[running_subprocess.parent_index]['max_childs'] = 0

        render_log()

    # 오류 코드를 반환했다면
    if returncode != 0 and throw:
        raise Exception(f'프로세스가 {returncode} 코드를 반환했습니다')

    return output, returncode

def runs(item: Union[Callable, List[Callable]]) -> bool:
    # 이게 다 파이썬이 익명 함수 지원 안해서 그런거임
    # 심플리티 뭐시기 ㅇㅈㄹ하면서 멀티 라인 없는 람다만 쓰게 강요하니까 이런거...
    # Pythonic 좆까 ㅗㅗ

    # 함수가 True 를 반환한다면 현재 단 작업 중단하기
    if callable(item):
        return item()
    elif isinstance(item, list):
        for child in item:
            if runs(child) == True:
                break
    else:
        # YOU FUCKING MORON
        raise('?')

# ==============================
# 작업 경로
# ==============================
def chdir(cwd: str) -> None:
    global LOG_FILE

    # 작업 경로 변경
    old_cwd = os.path.abspath(os.curdir)
    os.makedirs(cwd, exist_ok=True)
    os.chdir(cwd)

    # 필수 디렉터리 만들기
    os.makedirs('embeddings', exist_ok=True)
    os.makedirs('localizations', exist_ok=True)
    os.makedirs('scripts', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # 기존 로그 파일이 옮기기
    log_path = os.path.join('logs', datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M-%S.log'))

    if LOG_FILE:
        LOG_FILE.close()
        shutil.move(os.path.join(old_cwd, LOG_FILE.name), log_path)

    LOG_FILE = open(log_path, 'a')

    #  덮어쓸 설정 파일 가져오기
    if os.path.isfile(os.path.join(cwd, 'override.json')):
        log('설정 파일이 존재합니다, 기존 설정을 덮어씁니다')

        with open(os.path.join(cwd, 'override.json'), 'r') as file:
            override_options = json.loads(file.read())
            for key, value in override_options.items():
                if key not in OPTIONS:
                    log(f'{key} 키은 존재하는 설정 키가 아닙니다', styles={'color':'red'})
                    continue

                if type(value) != type(OPTIONS[key]):
                    log(f'{key} 키는 {type(OPTIONS[key]).__name__} 자료형이여만 합니다', styles={'color':'red'})
                    continue

                OPTIONS[key] = value

def has_python_package(pkg: str, check_loader=True) -> bool:
    spec = find_spec(pkg)
    return spec and (check_loader and spec.loader is not None)


# ==============================
# 파일 다운로드
# ==============================
def download(url: str, target=''):
    # 구글 드라이브 주소라면 gdown 패키지를 통해 가져오기
    if url.startswith('https://drive.google.com'):
        # 코랩 속에서만 패키지 받아오기
        if find_executable('gdown') is None:
            if IN_COLAB:
                execute(['pip', 'install', 'gdown'])
            else:
                raise('gdown 이 존재하지 않아 구글 드라이브로부터 파일을 받아올 수 없습니다')

        execute(['gdown', '-O', target, url])
        return

    # anonfile CDN 주소 가져오기
    if url.startswith('https://anonfiles.com/'):
        matches = re.search('https://cdn-[^\"]+', requests.get(url).text)
        if not matches:
            raise Exception('anonfiles 에서 CDN 주소를 파싱하는데 실패했습니다')

        url = matches[0]

    if os.path.isdir(target) or target.endswith('/'):
        # 목표 경로가 디렉터리라면
        dirname = target
        basename = ''
    else:
        # 목표 경로가 파일이거나 아예 존재하지 않다면
        dirname = os.path.dirname(target)
        basename = os.path.basename(target)

    # 목표 디렉터리 만들기
    if dirname != '':
        os.makedirs(dirname, exist_ok=True)

    if IN_COLAB and not find_executable('aria2c'):
        execute(['apt', 'install', 'aria2'], summary='빠른 다운로드를 위해 aria2 패키지를 설치합니다')

    if find_executable('aria2c'):
        args = [
            '--continue',
            '--allow-overwrite',
            '--always-resume',
            '--summary-interval=10',
            '--disk-cache=64M',
            '--min-split-size=8M',
            '--max-concurrent-downloads=8',
            '--max-connection-per-server=8',
            '--max-overall-download-limit=0',
            '--max-download-limit=0',
            '--split=8',
        ]

        if dirname != '':
            args.append(f'--dir={dirname}')

        if basename != '':
            # 목표 경로가 파일이거나 아예 존재하지 않다면
            args.append(f'--out={basename}')

        execute(['aria2c', *args, url], hide_summary=True)

    elif find_executable('curl'):
        args = ['--location']

        if basename == '':
            args += ['--remote-header-name', '--remote-name']
        else:
            args += ['--output', basename]

        execute(
            ['curl', *args, url], 
            hide_summary=True,
            cwd=dirname if dirname != '' else None
        )

    else:
        # 다른 패키지에선 파일 경로를 자동으로 잡아주는데 여기선 그럴 수 없으니 직접 해줘야됨
        # TODO: content-disposition 헤더로부터 파일 이름 가져오기
        if basename == '':
            basename = url.split('/')[-1]

        with requests.get(url, stream=True) as res:
            res.raise_for_status()
            with open(os.path.join(dirname, basename), 'wb') as file:
                # 받아온 파일 디코딩하기
                # https://github.com/psf/requests/issues/2155#issuecomment-50771010
                import functools
                res.raw.read = functools.partial(res.raw.read, decode_content=True)

                # TODO: 파일 길이가 적합한지?
                shutil.copyfileobj(res.raw, file, length=16*1024*1024)

def has_checkpoint() -> bool:
    for p in Path(os.path.join('models', 'Stable-diffusion')).glob('**/*.ckpt'):
        # aria2 로 받다만 파일은 무시하기
        if os.path.isfile(f'{p}.aria2'):
            continue

        return True
    return False


# ==============================
# WebUI 레포지토리 및 종속 패키지 설치
# ==============================
def patch_webui_repository() -> None:
    # 기본 UI 설정 값 (ui-config.json)
    # 설정 파일 자체를 덮어씌우면 새로 추가된 키를 인식하지 못해서 코드 자체를 수정함
    # https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/shared.py
    if os.path.isfile('repository/modules/shared.py'):
        log('설정 파일의 기본 값을 추천 값으로 변경합니다')
        cwd = os.path.abspath(os.curdir)

        configs = {
            # 기본 언어 파일
            'localization': 'ko-KR',

            # 결과 이미지 디렉터리
            'outdir_txt2img_samples': os.path.join(cwd, 'outputs', 'txt2img-samples'),
            'outdir_img2img_samples': os.path.join(cwd, 'outputs', 'img2img-samples'),
            'outdir_extras_samples': os.path.join(cwd, 'outputs', 'extras-samples'),
            'outdir_txt2img_grids': os.path.join(cwd, 'outputs', 'txt2img-grids'),
            'outdir_img2img_grids': os.path.join(cwd, 'outputs', 'img2img-grids'),

            # NAI 기본 설정(?)
            'CLIP_stop_at_last_layers': 2,
        }

        with open('repository/modules/shared.py', 'r+') as f:
            def replace(m: re.Match) -> str:
                if m[2] in configs:
                    # log(f'{m[2]} -> {configs[m[2]]}')
                    return f'{m[1]}{configs[m[2]]}{m[3]}'
                return m[0]

            # 기존 소스에서 설정 기본 값만 치환하기
            # '{key}': OptionInfo({value},
            replaced_code = re.sub(
                rf'(["\'](\w+)["\']:\s+?OptionInfo\(["\']?).+?(["\']?,)', 
                replace,
                f.read()
            )

            # 기존 내용 지우고 저장
            f.seek(0)
            f.truncate()
            f.write(replaced_code)

    # 기본 설정 파일 (config.json)
    if not os.path.isfile('ui-config.json'):
        log('UI 설정 파일이 존재하지 않습니다, 추천 값으로 새로 생성합니다')

        with open('ui-config.json', 'w') as f:
            configs = {
                'txt2img/Prompt/value': 'best quality, masterpiece',
                'txt2img/Negative prompt/value': 'lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry',
                'txt2img/Sampling Steps/value': 28,
                'txt2img/Width/value': 512,
                'txt2img/Height/value': 768,
                'txt2img/CFG Scale/value': 12,
            }

            f.write(json.dumps(configs, indent=4))

    # 스크립트 다운로드
    if OPTIONS['DOWNLOAD_CUSTOM_SCRIPTS']:
        log('사용자 스크립트를 받습니다')
        runs(PREDEFINED_SCRIPTS)

    # 확장 기능 다운로드
    if OPTIONS['DOWNLOAD_EXTENSIONS']:
        log('확장 기능을 받습니다')
        for ext in PREDEFINED_EXTENSIONS:
            exts_path = os.path.join(os.curdir, 'repository', 'extensions')
            ext_name = ext.split('/').pop().rstrip('.git')
            ext_path = os.path.join(exts_path, ext_name)
            
            if os.path.isdir(ext_path):
                execute(
                    ['git', 'pull'],
                    cwd=ext_path,
                    throw=False,
                    hide_summary=True
                )
            else:
                execute(
                    ['git', 'clone', '--recurse-submodules', '--depth', '1', ext],
                    cwd=exts_path,
                    throw=False,
                    hide_summary=True
                )

    syms = [
        'extensions',
        'javascript',
        'modules',
        'scripts',
        'textual_inversion_templates'
    ]

    for sym in syms:
        if not os.path.isdir(sym):
            continue

        log(f'{sym} 심볼릭 링크를 만듭니다')
    
        for path in os.listdir(sym):
            src = os.path.join(sym, path)
            dst = os.path.join('repository', sym, path)

            # 이미 파일 또는 디렉터리가 존재한다면 제거하기
            if os.path.exists(dst):
                if os.path.isfile(dst) or os.path.islink(dst):
                    os.remove(dst)
                else:
                    shutil.rmtree(dst, ignore_errors=True)

            # 심볼릭 링크 생성
            os.symlink(src, dst, target_is_directory=os.path.isdir(src))


def setup_webui() -> None:
    need_clone = True

    # 이미 디렉터리가 존재한다면 정상적인 레포인지 확인하기
    if os.path.isdir('repository'):
        try:
            # 사용자 파일만 남겨두고 레포지토리 초기화하기
            # https://stackoverflow.com/a/12096327
            execute(
                'git checkout -- . && git pull',
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
        path='repository'
        if os.path.islink(path):
            path = os.path.realpath(path)

        shutil.rmtree(path, ignore_errors=True)
        execute(
            ['git', 'clone', OPTIONS['REPO_URL'], path],
            summary='레포지토리를 가져옵니다'
        )

    # 특정 커밋이 지정됐다면 체크아웃하기
    if OPTIONS['REPO_COMMIT'] != '':
        execute(
            ['git', 'checkout', OPTIONS['REPO_COMMIT']],
            summary=f"레포지토리를 {OPTIONS['REPO_COMMIT']} 커밋으로 되돌립니다",
            cwd='repository'
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

    if line == 'Invalid ngrok authtoken, ngrok connection aborted.\n':
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
    matches = re.search('https?://[0-9a-f-]+\.(gradio\.app|([a-z]{2}\.)?ngrok\.io)', line)
    if matches:
        url = matches[0]

        # gradio 는 웹 서버가 켜진 이후 바로 나오기 때문에 사용자에게 바로 보여줘도 상관 없음
        if 'gradio.app' in url:
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
        if 'ngrok.io' in matches[0]:
            NGROK_URL = url

        return

def start_webui(args: List[str]=None, env: Dict[str, str]=None) -> None:
    global running_subprocess
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
        cwd = os.path.abspath(os.curdir)
        args = [
            '--ckpt-dir', os.path.join(cwd, 'models', 'Stable-diffusion'),
            '--embeddings-dir', os.path.join(cwd, 'embeddings'),
            '--hypernetwork-dir', os.path.join(cwd, 'models', 'hypernetworks'),
            '--codeformer-models-path', os.path.join(cwd, 'models', 'Codeformer'),
            '--gfpgan-models-path', os.path.join(cwd, 'models', 'GFPGAN'),
            '--esrgan-models-path', os.path.join(cwd, 'models', 'ESRGAN'),
            '--bsrgan-models-path', os.path.join(cwd, 'models', 'BSRGAN'),
            '--realesrgan-models-path', os.path.join(cwd, 'models', 'RealESRGAN'),
            '--scunet-models-path', os.path.join(cwd, 'models', 'ScuNET'),
            '--swinir-models-path', os.path.join(cwd, 'models', 'SwinIR'),
            '--ldsr-models-path', os.path.join(cwd, 'models', 'LDSR'),

            '--styles-file', os.path.join(cwd, 'styles.csv'),
            '--ui-config-file', os.path.join(cwd, 'ui-config.json'),
            '--ui-settings-file', os.path.join(cwd, 'config.json'),

            # TODO: 기븐으로 설정 해둬도 괜찮을까...?
            '--gradio-img2img-tool', 'color-sketch',
        ]

        # 기본 언어 파일 우선으로 사용하기
        if os.path.isdir(os.path.join(cwd, 'localizations')):
            args += ['--localizations-dir', os.path.join(cwd, 'localizations')]

        # 코랩 환경에선 메모리가 낮아 모델을 VRAM 위로 올려 사용해야함
        if IN_COLAB:
            args.append('--lowram')

        # xformers
        if OPTIONS['USE_XFORMERS']:
            log('xformers 를 사용합니다')

            if has_python_package('xformers'):
                args.append('--xformers')

            elif IN_COLAB:
                log('xformers 패키지가 존재하지 않습니다, 미리 컴파일된 xformers 패키지를 가져옵니다')
                download('https://github.com/toriato/easy-stable-diffusion/raw/prebuilt-xformers/cu113/xformers-0.0.14.dev0-cp37-cp37m-linux_x86_64.whl', tempfile.gettempdir())
                execute(
                    ['pip', 'install', os.path.join(tempfile.gettempdir(), 'xformers-0.0.14.dev0-cp37-cp37m-linux_x86_64.whl')],
                    summary='xformers 패키지를 설치합니다'
                )
                args.append('--xformers')

            else:
                # TODO: 패키지 빌드
                log('xformers 패키지가 존재하지 않습니다, --xformers 인자를 사용하지 않습니다')

        # deepdanbooru
        if OPTIONS['USE_DEEPBOORU']:
            log('deepbooru 를 사용합니다')
            args.append('--deepdanbooru')

        # gradio
        if OPTIONS['USE_GRADIO']:
            log('Gradio 터널을 사용합니다')
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
                OPTIONS['GRADIO_USERNAME'] + ('' if OPTIONS['GRADIO_PASSWORD'] == '' else ':' + OPTIONS['GRADIO_PASSWORD'])
            ]

        # ngrok
        if OPTIONS['NGROK_API_TOKEN'] != '':
            log('ngrok 터널을 사용합니다')
            args += [
                '--ngrok', OPTIONS['NGROK_API_TOKEN'],
                '--ngrok-region', 'jp'
            ]

            if has_python_package('pyngrok') is None:
                log('ngrok 사용에 필요한 패키지가 존재하지 않습니다, 설치를 시작합니다')
                execute(['pip', 'install', 'pyngrok'])

            # 추가 인자
            # TODO: 받은 문자열을 리스트로 안나누고 그대로 사용할 수 있는지?
            if OPTIONS['EXTRA_ARGS'] != '':
                args.append(OPTIONS['EXTRA_ARGS'])

    execute(
        ['python', 'launch.py', *args],
        parser=parse_webui_output,
        cwd='repository',
        env=env
    )


# ==============================
# 자 드게제~
# ==============================
try:
    # 코랩 폼 입력 란을 생성을 위한 코드
    # log(', '.join(map(lambda s:f'"{s}"', CHECKPOINTS.keys())))
    # raise

    # 인터페이스 출력
    if 'ipykernel' in sys.modules:
        from IPython.display import display
        from ipywidgets import widgets

        LOG_WIDGET = widgets.HTML()
        LOG_WIDGET.blocks = []

        display(LOG_WIDGET)

    log(platform.platform())
    log(f'Python {platform.python_version()}')
    log('')

    IN_COLAB = has_python_package('google') and has_python_package('google.colab')

    # 구글 드라이브 마운팅 시도
    if IN_COLAB and OPTIONS['USE_GOOGLE_DRIVE']:
        log('구글 드라이브 마운트를 시도합니다')

        from google.colab import drive
        drive.mount('/content/drive')

        # 구글 드라이브 기준
        chdir(os.path.join('/content/drive/MyDrive', OPTIONS['WORKSPACE']))
    else:
        chdir(os.path.join(os.path.abspath(os.curdir), OPTIONS['WORKSPACE']))

    # 코랩 환경 설정
    if IN_COLAB:
        log('코랩을 사용하고 있습니다')

        # 터널링 서비스가 아예 존재하지 않다면 오류 반환하기
        assert OPTIONS['USE_GRADIO'] or OPTIONS['NGROK_API_TOKEN'] != '', '터널링 서비스를 하나 이상 선택해주세요' 

        # 코랩 환경인데 글카가 왜 없어...?
        import torch
        assert torch.cuda.is_available(), 'GPU 가 없습니다, 런타임 유형이 잘못됐거나 GPU 할당량이 초과된 것 같습니다'

        # 코랩 환경에서 이유는 알 수 없지만 /usr 디렉터리 내에서 읽기/쓰기 속도가 다른 곳보다 월등히 빠름
        # 아마 /content 에 큰 용량을 박아두는 사용하는 사람들이 많아서 그런듯...?
        os.remove('repository') if os.path.islink('repository') else shutil.rmtree('repository', ignore_errors=True)
        os.symlink('/usr/local/repository', 'repository')

        # huggingface 모델 캐시 심볼릭 만들기
        # TODO: 다운로드 받는거보다 캐시가 더 빠른지 확인할 필요가 있음
        src = os.path.abspath(os.path.join('cache', 'huggingface'))
        dst = '/root/.cache/huggingface'
        os.remove(dst) if os.path.islink(dst) else shutil.rmtree(dst, ignore_errors=True)
        os.makedirs(src, exist_ok=True)
        os.symlink(src, dst, target_is_directory=True)

    # 체크포인트가 선택 존재한다면 해당 체크포인트 받기
    if not has_checkpoint():
        log('체크포인트가 존재하지 않습니다, 자동으로 추천 체크포인트를 받아옵니다')

        for file in DEFAULT_CHECKPOINT_URLS:
            target = os.path.join('models', 'Stable-diffusion', file.get('target', ''))
            download(**{**file, 'target': target})


    # WebUI 가져오기
    setup_webui()

    # WebUI 실행
    start_webui()

# ^c 종료 무시하기
except KeyboardInterrupt:
    pass

except:
    # 로그 위젯이 없다면 평범하게 오류 처리하기
    if not LOG_WIDGET:
        raise

    log_trace()
