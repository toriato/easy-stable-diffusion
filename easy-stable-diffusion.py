# @title
# fmt: off
import os
import shutil
import sys
import platform
import re
import json
import requests
from typing import Union, Callable, Tuple, List
from subprocess import Popen, PIPE, STDOUT
from distutils.spawn import find_executable
from importlib.util import find_spec
from pathlib import Path
from datetime import datetime

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

def append_log_block(summary: str='', lines: List[str]=None,
                     summary_styles: dict=None, line_styles: dict=None,
                     max_lines=0) -> int:
    block = {**locals()}

    # 인자에 직접 기본 값을 넣으면 값을 돌려쓰기 때문에 직접 생성해줘야됨
    if block['lines'] is None: block['lines'] = []
    if block['summary_styles'] is None: block['summary_styles'] = {}
    if block['line_styles'] is None: block['line_styles'] = {
        'padding-left': '1.5em',
        'color': 'gray'
    }

    LOG_WIDGET.blocks.append(block)
    return len(LOG_WIDGET.blocks) - 1

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
        summary_styles = {
            'display': 'inline-block',
            **block['summary_styles']
        }
        line_styles = {
            'display': 'inline-block',
            **block['line_styles']
        }

        html += f'<span style="{format_styles(summary_styles)}">{block["summary"]}</span>\n'

        if block['max_lines'] is not None and len(block['lines']) > 0:
            html += f'<div style="{format_styles(line_styles)}">'
            html += ''.join(block['lines'][-block['max_lines']:])
            html += '</div>'

    html += '</div>'

    LOG_WIDGET.value = html

def log(msg: str, styles={}, newline=True, block_index: int=None,
        print_to_file=True, print_to_widget=True) -> None:
    # 기록할 내용이 ngrok API 키와 일치한다면 숨기기
    # TODO: 더 나은 문자열 검사, 원치 않은 내용이 가려질 수도 있음
    if NGROK_API_TOKEN != '':
        msg = msg.replace(NGROK_API_TOKEN, '**REDACTED**')

    if newline:
        msg += '\n'

    # 파일에 기록하기
    if print_to_file:
        if LOG_FILE:
            if block_index and msg.endswith('\n'):
                LOG_FILE.write('\t')
            LOG_FILE.write(msg)
            LOG_FILE.flush()

    # 로그 위젯이 존재한다면 위젯에 표시하기
    if print_to_widget and LOG_WIDGET:
        if block_index is None:
            block_index = append_log_block(summary=msg, summary_styles=styles)
        else:
            LOG_WIDGET.blocks[block_index]['lines'].append(msg)
        render_log()
        return

    print(msg, end='')

def log_trace() -> None:
    import traceback

    # 스택 가져오기
    ex_type, ex_value, ex_traceback = sys.exc_info()

    summary_styles = {}

    # 오류가 존재한다면 메세지 빨간색으로 출력하기
    # https://docs.python.org/3/library/sys.html#sys.exc_info
    # TODO: 오류 유무 이렇게 확인하면 안될거 같은데 일단 귀찮아서 대충 써둠
    if ex_type is not None and 'color' not in summary_styles:
        summary_styles = LOG_WIDGET_STYLES['dialog_error']

    block_index = None if LOG_WIDGET is None else append_log_block(
        summary='보고서를 만들고 있습니다...', 
        summary_styles=summary_styles
    )

    # 오류가 존재한다면 오류 정보와 스택 트레이스 출력하기
    if ex_type is not None:
        log(block_index=block_index, msg=f'{ex_type.__name__}: {ex_value}')
        log(
            block_index=block_index, 
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
        # 이전 로그 싹 긁어오기
        logs = ''   
        with open(LOG_FILE.name) as file:
            logs = file.read()

        # 로그 업로드
        # TODO: 업로드 실패 시 오류 처리
        res = requests.post('https://hastebin.com/documents', data=logs.encode('utf-8'))
        url = f"https://hastebin.com/{json.loads(res.text)['key']}"

        # 기존 오류 메세지 업데이트
        LOG_WIDGET.blocks[block_index]['summary'] = '\n'.join([
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

    if LOG_WIDGET:
        running_subprocess.block_index = append_log_block(
            f'=> {summary}\n',
            summary_styles={ 'color': 'yellow' },
            max_lines = 5,
        )
    else:
        running_subprocess.block_index = None
        log(f'=> {summary}')

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
            block_index=running_subprocess.block_index, 
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
                del LOG_WIDGET.blocks[running_subprocess.block_index]
            else:
                LOG_WIDGET.blocks[running_subprocess.block_index]['summary_styles']['color'] = 'green'
                LOG_WIDGET.blocks[running_subprocess.block_index]['max_lines'] = None

        else:
            LOG_WIDGET.blocks[running_subprocess.block_index]['summary_styles']['color'] = 'red'
            LOG_WIDGET.blocks[running_subprocess.block_index]['max_lines'] = 0

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
PATHS = {}

def update_path_to(path_to_workspace: str) -> None:
    global LOG_FILE

    PATHS['workspace'] = path_to_workspace
    PATHS['outputs'] = f"{PATHS['workspace']}/outputs"
    PATHS['models'] = f"{PATHS['workspace']}/models"
    PATHS['embeddings'] = f"{PATHS['workspace']}/embeddings"
    PATHS['localizations'] = f"{PATHS['workspace']}/localizations"
    PATHS['scripts'] = f"{PATHS['workspace']}/scripts"
    PATHS['logs'] = f"{PATHS['workspace']}/logs"
    PATHS['styles_file'] = f"{PATHS['workspace']}/styles.csv"
    PATHS['ui_config_file'] = f"{PATHS['workspace']}/ui-config.json"
    PATHS['ui_settings_file'] = f"{PATHS['workspace']}/config.json"

    os.makedirs(PATHS['workspace'], exist_ok=True)
    os.makedirs(PATHS['embeddings'], exist_ok=True)
    os.makedirs(PATHS['localizations'], exist_ok=True)
    os.makedirs(PATHS['scripts'], exist_ok=True)
    os.makedirs(PATHS['logs'], exist_ok=True)

    log_path = os.path.join(PATHS['logs'], datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M-%S.log'))

    # 기존 로그 파일이 존재한다면 옮기기
    if LOG_FILE:
        LOG_FILE.close()
        shutil.move(LOG_FILE.name, log_path)

    LOG_FILE = open(log_path, 'a')

def has_python_package(pkg: str, check_loader=True) -> bool:
    spec = find_spec(pkg)
    return spec and (check_loader and spec.loader is not None)

# ==============================
# 사용자 설정
# ==============================
CHECKPOINTS = {
    # NAI leaks
    'NAI - animefull-final-pruned': {
        'files': [
            {
                'url': 'https://pub-2fdef7a2969f43289c42ac5ae3412fd4.r2.dev/animefull-final-pruned.ckpt',
                'target': 'nai/animefull-final-pruned.ckpt',
            },
            {
                'url': 'https://pub-2fdef7a2969f43289c42ac5ae3412fd4.r2.dev/animevae.pt',
                'target': 'nai/animefull-final-pruned.vae.pt'
            },
            {
                'url': 'https://gist.githubusercontent.com/toriato/ae1f587f4d1e9ee5d0e910c627277930/raw/6019f8782875497f6e5b3e537e30a75df5b64812/animefull-final-pruned.yaml',
                'target': 'nai/animefull-final-pruned.yaml'
            }
        ]
    },
    'NAI - animefull-latest': {
        'files': [
            {
                'url': 'https://pub-2fdef7a2969f43289c42ac5ae3412fd4.r2.dev/animefull-latest.ckpt',
                'target': 'nai/animefull-latest.ckpt'
            },
            {
                'url': 'https://pub-2fdef7a2969f43289c42ac5ae3412fd4.r2.dev/animevae.pt',
                'target': 'nai/animefull-latest.vae.pt'
            },
            {
                'url': 'https://gist.githubusercontent.com/toriato/ae1f587f4d1e9ee5d0e910c627277930/raw/6019f8782875497f6e5b3e537e30a75df5b64812/animefull-latest.yaml',
                'target': 'nai/animefull-latest.yaml'
            }
        ]
    },
    'NAI - animesfw-final-pruned': {
        'files': [
            {
                'url': 'https://pub-2fdef7a2969f43289c42ac5ae3412fd4.r2.dev/animesfw-final-pruned.ckpt',
                'target': 'nai/animesfw-final-pruned.ckpt'
            },
            {
                'url': 'https://pub-2fdef7a2969f43289c42ac5ae3412fd4.r2.dev/animevae.pt',
                'target': 'nai/animesfw-final-pruned.vae.pt'
            },
            {
                'url': 'https://gist.github.com/toriato/ae1f587f4d1e9ee5d0e910c627277930/raw/6019f8782875497f6e5b3e537e30a75df5b64812/animesfw-final-pruned.yaml',
                'target': 'nai/animesfw-final-pruned.yaml'
            }
        ]
    },
    'NAI - animesfw-latest': {
        'files': [
            {
                'url': 'https://pub-2fdef7a2969f43289c42ac5ae3412fd4.r2.dev/animesfw-latest.ckpt',
                'target': 'nai/animesfw-latest.ckpt'
            },
            {
                'url': 'https://pub-2fdef7a2969f43289c42ac5ae3412fd4.r2.dev/animevae.pt',
                'target': 'nai/animesfw-latest.vae.pt'
            },
            {
                'url': 'https://gist.github.com/toriato/ae1f587f4d1e9ee5d0e910c627277930/raw/6019f8782875497f6e5b3e537e30a75df5b64812/animesfw-latest.yaml',
                'target': 'nai/animesfw-latest.yaml'
            }
        ]
    },

    # Waifu Diffusion
    'Waifu Diffusion 1.3': {
        'files': [{
            'url': 'https://huggingface.co/hakurei/waifu-diffusion-v1-3/resolve/main/wd-v1-3-float16.ckpt',
            'args': ['-o', 'wd-v1-3-epoch09-float16.ckpt']
        }]
    },

    # Trinart2
    'Trinart Stable Diffusion v2 60,000 Steps': {
        'files': [{'url': 'https://huggingface.co/naclbit/trinart_stable_diffusion_v2/resolve/main/trinart2_step60000.ckpt'}]
    },
    'Trinart Stable Diffusion v2 95,000 Steps': {
        'files': [{'url': 'https://huggingface.co/naclbit/trinart_stable_diffusion_v2/resolve/main/trinart2_step95000.ckpt'}]
    },
    'Trinart Stable Diffusion v2 115,000 Steps': {
        'files': [{'url': 'https://huggingface.co/naclbit/trinart_stable_diffusion_v2/resolve/main/trinart2_step115000.ckpt'}]
    },

    'Furry (epoch 4)': {
        'files': [{'url': 'https://iwiftp.yerf.org/Furry/Software/Stable%20Diffusion%20Furry%20Finetune%20Models/Finetune%20models/furry_epoch4.ckpt'}]
    },
    'Zack3D Kinky v1': {
        'files': [{'url': 'https://iwiftp.yerf.org/Furry/Software/Stable%20Diffusion%20Furry%20Finetune%20Models/Finetune%20models/Zack3D_Kinky-v1.ckpt'}]
    },
    'Pokemon': {
        'files': [{
            'url': 'https://huggingface.co/justinpinkney/pokemon-stable-diffusion/resolve/main/ema-only-epoch%3D000142.ckpt',
            'args': ['-o', 'pokemon-ema-pruned.ckpt']
        }]
    },
    'Dreambooth - Hiten': {
        'files': [{'url': 'https://huggingface.co/BumblingOrange/Hiten/resolve/main/Hiten%20girl_anime_8k_wallpaper_4k.ckpt'}]
    },
}

# @markdown ### <font color="orange">***다운로드 받을 모델(체크포인트) 선택***</font>
# @markdown 입력 란을 <font color="red">비워두면</font> 모델을 받지 않고 바로 실행함
# @markdown <br>우측 <font color="red">화살표(🔽)</font> 클릭하면 모델 선택 가능
CHECKPOINT = '' #@param ["", "NAI - animefull-final-pruned", "NAI - animefull-latest", "NAI - animesfw-final-pruned", "NAI - animesfw-latest", "Waifu Diffusion 1.3", "Trinart Stable Diffusion v2 60,000 Steps", "Trinart Stable Diffusion v2 95,000 Steps", "Trinart Stable Diffusion v2 115,000 Steps", "Furry (epoch 4)", "Zack3D Kinky v1", "Pokemon", "Dreambooth - Hiten"] {allow-input: true}

# @markdown ### <font color="orange">***구글 드라이브 동기화를 사용할지?***</font>
USE_GOOGLE_DRIVE = True  # @param {type:"boolean"}

# @markdown ### <font color="orange">***구글 드라이브 작업 디렉터리 경로***</font>
# @markdown 임베딩, 모델, 결과, 설정 등 영구적으로 보관될 파일이 저장될 디렉터리의 경로
PATH_TO_GOOGLE_DRIVE = 'SD' # @param {type:"string"}

# @markdown ##### <font size="2" color="red">(선택)</font> <font color="orange">***xformers 를 사용할지?***</font>
# @markdown - <font color="green">장점</font>: 성능 향생
# @markdown - <font color="red">단점</font>: 미리 빌드한 패키지가 지원하지 않는 환경에선 직접 빌드할 필요가 있음
USE_XFORMERS = True  # @param {type:"boolean"}

# @markdown ##### <font size="2" color="red">(선택)</font> <font color="orange">***deepbooru 를 사용할지?***</font>
# @markdown IMG2IMG 에 올린 이미지를 단부루 태그로 변환(예측)해 프롬프트로 추출해내는 기능
# @markdown - <font color="red">단점</font>: 처음 실행할 때 추가 패키지를 받기 때문에 시간이 조금 더 걸림
USE_DEEPDANBOORU = True  # @param {type:"boolean"}

# @markdown ##### <font size="2" color="red">(선택)</font> <font color="orange">***Gradio 터널을 사용할지?***</font>
USE_GRADIO_TUNNEL = True # @param {type:"boolean"}

# @markdown ##### <font size="2" color="red">(선택)</font> <font color="orange">***Gradio 인증 정보***</font>
# @markdown Gradio 접속 시 사용할 사용자 아이디와 비밀번호
# @markdown <br>`GRADIO_USERNAME` 입력 란을 <font color="red">비워두면</font> 인증을 사용하지 않음
# @markdown <br>`GRADIO_USERNAME` 입력 란에 `user1:pass1,user,pass2`처럼 입력하면 여러 사용자 추가 가능
# @markdown <br>`GRADIO_PASSWORD` 입력 란을 <font color="red">비워두면</font> 자동으로 비밀번호를 생성함
GRADIO_USERNAME = 'gradio' # @param {type:"string"}
GRADIO_PASSWORD = '' # @param {type:"string"}
GRADIO_PASSWORD_GENERATED = False

# @markdown ##### <font size="2" color="red">(선택)</font> <font color="orange">***ngrok API 키***</font>
# @markdown ngrok 터널에 사용할 API 토큰
# @markdown <br>[API 토큰은 여기를 눌러 계정을 만든 뒤 얻을 수 있음](https://dashboard.ngrok.com/get-started/your-authtoken)
# @markdown <br>입력 란을 <font color="red">비워두면</font> ngrok 터널을 비활성화함
NGROK_API_TOKEN = '' # @param {type:"string"}
NGROK_URL = None

# @markdown ##### <font size="2" color="red">(선택)</font> <font color="orange">***WebUI 레포지토리 주소***</font>
REPO_URL = 'https://github.com/AUTOMATIC1111/stable-diffusion-webui.git' # @param {type:"string"}

# @markdown ##### <font size="2" color="red">(선택)</font> <font color="orange">***WebUI 레포지토리 커밋 해시***</font>
# @markdown 입력 란을 <font color="red">비워두면</font> 가장 최신 커밋을 가져옴
REPO_COMMIT = '' # @param {type:"string"}

# 레포지토리에 적용할 풀 리퀘스트
REPO_PULL_REQUESTS = []

# 추가로 받을 스크립트
ADDITIONAL_SCRIPTS = [
    # 번역 파일
    lambda: download(
        'https://gist.github.com/toriato/72847da83f44d8d9d1eb6b0027fc329f/raw/a7e9896d9a796d20ec55077b81d32cc5155e83ab/ko-KR_easy-stable-diffusion.json',
        PATHS['localizations'],
    ),

    # 태그 자동 완성 유저스크립트
    # https://arca.live/b/aiart/60536925/272094058
    lambda: download(
        'https://greasyfork.org/scripts/452929-webui-%ED%83%9C%EA%B7%B8-%EC%9E%90%EB%8F%99%EC%99%84%EC%84%B1/code/WebUI%20%ED%83%9C%EA%B7%B8%20%EC%9E%90%EB%8F%99%EC%99%84%EC%84%B1.user.js',
        'repo/javascript',
    ),

    # Advanced prompt matrix
    # https://github.com/GRMrGecko/stable-diffusion-webui-automatic/blob/advanced_matrix/scripts/advanced_prompt_matrix.py
    lambda: download(
        'https://raw.githubusercontent.com/GRMrGecko/stable-diffusion-webui-automatic/advanced_matrix/scripts/advanced_prompt_matrix.py',
        'repo/scripts'
    ),

    # Dynamic Prompt Templates
    # https://github.com/adieyal/sd-dynamic-prompting
    lambda: download(
        'https://github.com/adieyal/sd-dynamic-prompting/raw/main/dynamic_prompting.py',
        'repo/scripts'
    ),

    # Wildcards
    # https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Custom-Scripts#wildcards
    [
        lambda: download(
            'https://raw.githubusercontent.com/jtkelm2/stable-diffusion-webui-1/master/scripts/wildcards.py',
            'repo/scripts'
        ),
        # 스크립트 디렉터리는 patch_webui_repository 메소드에서
        # 코랩 환경일 때 심볼릭 링크를 만들기 때문에 따로 처리할 필요가 없음
        [
            # 사용자 디렉터리가 존재하지 않는다면 기본 데이터셋 가져오기
            # https://github.com/Lopyter/stable-soup-prompts
            lambda: os.path.exists('repo/scripts/wildcards'), # True 반환시 현재 리스트 실행 정지
            lambda: shutil.rmtree('.tmp', ignore_errors=True),
            lambda: execute(
                ['git', 'clone', 'https://github.com/Lopyter/stable-soup-prompts.git', '.tmp'],
                hide_summary=True    
            ),
            lambda: os.remove('repo/scripts/wildcards') if os.path.islink('repo/scripts/wildcards') else None, # 심볼릭 링크는 파일로 삭제해야함
            lambda: shutil.rmtree('repo/scripts/wildcards', ignore_errors=True),
            lambda: shutil.copytree('.tmp/wildcards', 'repo/scripts/wildcards'),
            lambda: shutil.rmtree('.tmp', ignore_errors=True)
        ]
    ],

    # txt2mask
    # https://github.com/ThereforeGames/txt2mask
    [
        lambda: shutil.rmtree('.tmp', ignore_errors=True),
        lambda: execute(
            ['git', 'clone', 'https://github.com/ThereforeGames/txt2mask.git', '.tmp'],
            hide_summary=True
        ),
        lambda: shutil.rmtree('repo/repositories/clipseg', ignore_errors=True),
        lambda: shutil.copytree('.tmp/repositories/clipseg', 'repo/repositories/clipseg'),
        lambda: shutil.copy('.tmp/scripts/txt2mask.py', 'repo/scripts'),
        lambda: shutil.rmtree('.tmp', ignore_errors=True),
    ],

    # Img2img Video
    # https://github.com/memes-forever/Stable-diffusion-webui-video
    lambda: download(
        'https://raw.githubusercontent.com/memes-forever/Stable-diffusion-webui-video/main/videos.py',
        'repo/scripts'
    ),

    # Seed Travel
    # https://github.com/yownas/seed_travel
    [
        lambda: None if has_python_package('moviepy') else execute(['pip', 'install', 'moviepy']),
        lambda: download(
            'https://raw.githubusercontent.com/yownas/seed_travel/main/scripts/seed_travel.py',
            'repo/scripts',
        )
    ],

    # Animator
    # https://github.com/Animator-Anon/Animator
    lambda: download(
        'https://raw.githubusercontent.com/Animator-Anon/Animator/main/animation.py',
        'repo/scripts'
    ),

    # Alternate Noise Schedules
    # https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Custom-Scripts#alternate-noise-schedules
    lambda: download(
        'https://gist.githubusercontent.com/dfaker/f88aa62e3a14b559fe4e5f6b345db664/raw/791dabfa0ab26399aa2635bcbc1cf6267aa4ffc2/alternate_sampler_noise_schedules.py',
        'repo/scripts'
    ),

    # Vid2Vid
    # https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Custom-Scripts#vid2vid
    lambda: download(
        'https://raw.githubusercontent.com/Filarius/stable-diffusion-webui/master/scripts/vid2vid.py',
        'repo/scripts'
    ),

    # Shift Attention
    # https://github.com/yownas/shift-attention
    [
        lambda: None if has_python_package('moviepy') else execute(['pip', 'install', 'moviepy']),
        lambda: download(
            'https://raw.githubusercontent.com/yownas/shift-attention/main/scripts/shift_attention.py',
            'repo/scripts'
        )
    ],

    # Loopback and Superimpose
    # https://github.com/DiceOwl/StableDiffusionStuff
    lambda: download(
        'https://raw.githubusercontent.com/DiceOwl/StableDiffusionStuff/main/loopback_superimpose.py',
        'repo/scripts'
    ),

    # Run n times
    # https://gist.github.com/camenduru/9ec5f8141db9902e375967e93250860f
    lambda: download(
        'https://gist.githubusercontent.com/camenduru/9ec5f8141db9902e375967e93250860f/raw/b5c741676c5514105b9a1ea7dd438ca83802f16f/run_n_times.py',
        'repo/scripts'
    ),

    # Advanced Loopback
    # https://github.com/Extraltodeus/advanced-loopback-for-sd-webui
    lambda: download(
        'https://raw.githubusercontent.com/Extraltodeus/advanced-loopback-for-sd-webui/main/advanced_loopback.py',
        'repo/scripts'
    ),

    # prompt-morph
    # https://github.com/feffy380/prompt-morph
    [
        lambda: None if has_python_package('moviepy') else execute(['pip', 'install', 'moviepy']),
        lambda: download(
            'https://raw.githubusercontent.com/feffy380/prompt-morph/master/prompt_morph.py',
            'repo/scripts'
        ),
    ],

    # prompt interpolation
    # https://github.com/EugeoSynthesisThirtyTwo/prompt-interpolation-script-for-sd-webui
    lambda: download(
        'https://raw.githubusercontent.com/EugeoSynthesisThirtyTwo/prompt-interpolation-script-for-sd-webui/main/prompt_interpolation.py',
        'repo/scripts'
    ),

    # Asymmetric Tiling
    # https://github.com/tjm35/asymmetric-tiling-sd-webui/
    lambda: download(
        'https://raw.githubusercontent.com/tjm35/asymmetric-tiling-sd-webui/main/asymmetric_tiling.py',
        'repo/scripts'
    ),

    # Booru tag autocompletion for A1111
    # https://github.com/DominikDoom/a1111-sd-webui-tagcomplete
    [
        lambda: shutil.rmtree('.tmp', ignore_errors=True),
        lambda: execute(
            ['git', 'clone', 'https://github.com/DominikDoom/a1111-sd-webui-tagcomplete.git', '.tmp'],
            hide_summary=True
        ),
        [
            # 코랩 + 사용자 디렉터리가 존재한다면 심볼릭 링크 만들기
            lambda: not (IN_COLAB and os.path.isdir(os.path.join(PATHS['workspace'], 'tags'))),  # True 반환시 현재 리스트 실행 정지
            lambda: shutil.rmtree('repo/tags', ignore_errors=True),
            lambda: os.symlink('repo/tags', os.path.join(PATHS['workspace'], 'tags'))
        ],
        [
            # 사용자 디렉터리가 존재하지 않는다면 기본 데이터셋 가져오기
            lambda: IN_COLAB and os.path.islink('repo/tags'),  # True 반환시 현재 리스트 실행 정지
            lambda: not IN_COLAB and os.path.isdir('repo/tags'),  # True 반환시 현재 리스트 실행 정지
            lambda: shutil.rmtree('repo/tags', ignore_errors=True),
            lambda: shutil.copytree('.tmp/tags', 'repo/tags'),
        ],
        lambda: shutil.copy('.tmp/javascript/tagAutocomplete.js', 'repo/javascript'),
        lambda: shutil.copy('.tmp/scripts/tag_autocomplete_helper.py', 'repo/scripts'),
        lambda: shutil.rmtree('.tmp', ignore_errors=True),
    ]
]

# @markdown ##### <font size="2" color="red">(선택)</font> <font color="orange">***WebUI 추가 인자***</font>
ADDITIONAL_ARGS = '' # @param {type:"string"}

# 로그 파일
LOG_FILE = None

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
IN_COLAB = has_python_package('google') and has_python_package('google.colab')

# ==============================
# 구글 드라이브 동기화
# ==============================
def mount_google_drive() -> None:
    log('구글 드라이브 마운트를 시작합니다')

    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)

    # 전체 경로 업데이트
    update_path_to(os.path.join('/content/drive/MyDrive', PATH_TO_GOOGLE_DRIVE))


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

def download_checkpoint(checkpoint: str) -> None:
    if checkpoint in CHECKPOINTS:
        checkpoint = CHECKPOINTS[checkpoint]
    else:
        # 미리 선언된 체크포인트가 아니라면 주소로써 사용하기
        checkpoint = {'files': [{'url': checkpoint}]}

    # Aria2 로 모델 받기
    # TODO: 토렌트 마그넷 주소 지원
    log(f"파일 {len(checkpoint['files'])}개를 받습니다")

    for file in checkpoint['files']:
        target = os.path.join(f"{PATHS['models']}/Stable-diffusion", file.get('target', ''))
        download(**{**file, 'target': target})

def has_checkpoint() -> bool:
    for p in Path(f"{PATHS['models']}/Stable-diffusion").glob('**/*.ckpt'):
        # aria2 로 받다만 파일은 무시하기
        if os.path.isfile(f'{p}.aria2'):
            continue

        return True
    return False


# ==============================
# WebUI 레포지토리 및 종속 패키지 설치
# ==============================
def patch_webui_pull_request(number: int) -> None:
    res = requests.get(f'https://api.github.com/repos/AUTOMATIC1111/stable-diffusion-webui/pulls/{number}')
    payload = res.json()

    log(f"풀 리퀘스트 적용을 시도합니다: #{number} {payload['title']}")
    if payload['state'] != 'open':
        log(f'닫힌 풀 리퀘스트이므로 넘깁니다')
        return

    execute(f"curl -sSL {payload['patch_url']} | git apply", 
        throw=False,
        shell=True,
        cwd='repo'
    )

def patch_webui_repository() -> None:
    # 기본 UI 설정 값 (ui-config.json)
    # 설정 파일 자체를 덮어씌우면 새로 추가된 키를 인식하지 못해서 코드 자체를 수정함
    # https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/shared.py
    if os.path.isfile('repo/modules/shared.py'):
        log('설정 파일의 기본 값을 추천 값으로 변경합니다')

        configs = {
            # 기본 언어 파일
            'localization': os.path.join(PATHS['localizations'], 'ko-KR_easy-stable-diffusion.json'),

            # 결과 이미지 디렉터리
            'outdir_txt2img_samples': os.path.join(PATHS['outputs'], 'txt2img-samples'),
            'outdir_img2img_samples': os.path.join(PATHS['outputs'], 'img2img-samples'),
            'outdir_extras_samples': os.path.join(PATHS['outputs'], 'extras-samples'),
            'outdir_txt2img_grids': os.path.join(PATHS['outputs'], 'txt2img-grids'),
            'outdir_img2img_grids': os.path.join(PATHS['outputs'], 'img2img-grids'),

            # NAI 기본 설정(?)
            'CLIP_stop_at_last_layers': 2,
        }

        with open('repo/modules/shared.py', 'r+') as f:
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
    if not os.path.isfile(PATHS['ui_config_file']):
        log('UI 설정 파일이 존재하지 않습니다, 추천 값으로 새로 생성합니다')

        with open(PATHS['ui_config_file'], 'w') as f:
            configs = {
                'txt2img/Prompt/value': 'best quality, masterpiece',
                'txt2img/Negative prompt/value': 'lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry',
                'txt2img/Sampling Steps/value': 28,
                'txt2img/Width/value': 512,
                'txt2img/Height/value': 768,
                'txt2img/CFG Scale/value': 12,
            }

            f.write(json.dumps(configs, indent=4))

    # 풀 리퀘스트 적용
    if REPO_URL.startswith('https://github.com/AUTOMATIC1111/stable-diffusion-webui'):
        for number in REPO_PULL_REQUESTS:
            patch_webui_pull_request(number)

    # 스크립트 다운로드
    log('사용자 스크립트를 받습니다')
    runs(ADDITIONAL_SCRIPTS)

    # 사용자 스크립트 심볼릭 링크 생성
    log('사용자 스크립트의 심볼릭 링크를 만듭니다')
    for path in os.listdir(PATHS['scripts']):
        src = os.path.join(PATHS['scripts'], path)
        dst = os.path.join('repo/scripts', os.path.basename(path))

        # 이미 파일이 존재한다면 기존 파일 삭제하기
        if os.path.exists(dst):
            os.remove(dst) if os.path.islink(dst) else shutil.rmtree(dst, ignore_errors=True)

        # 심볼릭 링크 생성
        os.symlink(src, dst, target_is_directory=os.path.isdir(path))

def setup_webui() -> None:
    need_clone = True

    # 이미 디렉터리가 존재한다면 정상적인 레포인지 확인하기
    if os.path.isdir('repo'):
        try:
            # 사용자 파일만 남겨두고 레포지토리 초기화하기
            # https://stackoverflow.com/a/12096327
            execute(
                'git checkout -- . && git pull',
                summary='레포지토리를 업데이트 합니다',
                shell=True,
                cwd='repo'
            )

            need_clone = False

        except:
            log('레포지토리가 잘못됐습니다, 디렉터리를 제거합니다')

    if need_clone:
        shutil.rmtree('repo', ignore_errors=True)
        execute(
            ['git', 'clone', REPO_URL, 'repo'],
            summary='레포지토리를 가져옵니다'
        )

    # 특정 커밋이 지정됐다면 체크아웃하기
    if REPO_COMMIT != '':
        execute(
            ['git', 'checkout', REPO_COMMIT],
            summary=f'레포지토리를 {REPO_COMMIT} 커밋으로 되돌립니다'
        )

    patch_webui_repository()

def parse_webui_output(line: str) -> bool:
    global NGROK_URL

    # 하위 파이썬 실행 중 오류가 발생하면 전체 기록 표시하기
    # TODO: 더 나은 오류 핸들링, 잘못된 내용으로 트리거 될 수 있음
    if LOG_WIDGET and 'Traceback (most recent call last):' in line:
        LOG_WIDGET.blocks[running_subprocess.block_index]['max_lines'] = 0
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
                    'Gradio 비밀번호가 자동으로 생성됐습니다',
                    f'아이디: {GRADIO_USERNAME}',
                    f'비밀번호: {GRADIO_PASSWORD}'
                ]),
                LOG_WIDGET_STYLES['dialog_success'], 
                print_to_file=False
            )

        # ngork
        if NGROK_API_TOKEN != '':
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
    matches = re.search('https?://[0-9a-f-]+\.(gradio\.app|ngrok\.io)', line)
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

def start_webui(args: List[str]=[], env={}) -> None:
    global running_subprocess

    # 이미 WebUI 가 실행 중인지 확인하기
    # TODO: 비동기 없이 순차적으로 실행되는데 이 코드가 꼭 필요한지?
    if running_subprocess and running_subprocess.poll() is None:
        if 'launch.py' in running_subprocess.args:
            log('이미 실행 중인 웹UI를 종료하고 다시 시작합니다')
            running_subprocess.kill()

    execute(
        ['python', 'launch.py', *args],
        parser=parse_webui_output,
        cwd='repo',
        env={
            **os.environ,
            'PYTHONUNBUFFERED': '1',
            'REQS_FILE': 'requirements.txt',
            **env
        }
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

    # 기본 작업 경로 설정
    update_path_to(os.path.abspath(os.curdir))

    log(platform.platform())
    log(f'Python {platform.python_version()}')
    log('')

    if IN_COLAB:
        log('코랩을 사용하고 있습니다')

        assert USE_GRADIO_TUNNEL or NGROK_API_TOKEN != '', '터널링 서비스를 하나 이상 선택해주세요' 

        import torch
        assert torch.cuda.is_available(), 'GPU 가 없습니다, 런타임 유형이 잘못됐거나 GPU 할당량이 초과된 것 같습니다'

        # 구글 드라이브 마운팅 시도
        if USE_GOOGLE_DRIVE:
            mount_google_drive()

        # 코랩 환경에서 이유는 알 수 없지만 /usr 디렉터리 내에서 읽기/쓰기 속도가 다른 곳보다 월등히 빠름
        # 아마 /content 에 큰 용량을 박아두는 사용하는 사람들이 많아서 그런듯...?
        os.makedirs('/usr/local/content', exist_ok=True)
        os.chdir('/usr/local/content')

        # huggingface 모델 캐시 심볼릭 만들기
        dst = '/root/.cache/huggingface'

        if not os.path.islink(dst):
            log('트랜스포머 모델 캐시 디렉터리에 심볼릭 링크를 만듭니다')
            shutil.rmtree(dst, ignore_errors=True)

            src = os.path.join(PATHS['workspace'], 'cache', 'huggingface')
            os.makedirs(src, exist_ok=True)
            os.symlink(src, dst, target_is_directory=True)

    # 체크포인트가 선택 존재한다면 해당 체크포인트 받기
    if CHECKPOINT == '':
        if not has_checkpoint():
            if IN_COLAB:
                log('체크포인트가 존재하지 않습니다')
                log('추천 체크포인트를 자동으로 다운로드 합니다')
                download_checkpoint('NAI - animefull-final-pruned')
            else: 
                raise Exception('체크포인트가 존재하지 않습니다')
    else:
        log('선택한 체크포인트를 다운로드 합니다')
        log('다운로드 작업을 원치 않는다면 CHECKPOINT 옵션의 입력 란을 비워두고 다시 실행해주세요')
        download_checkpoint(CHECKPOINT)


    # WebUI 가져오기
    setup_webui()

    # WebUI 실행
    args = [
        # 동적 경로들
        '--ckpt-dir', f"{PATHS['models']}/Stable-diffusion",
        '--embeddings-dir', PATHS['embeddings'],
        '--hypernetwork-dir', f"{PATHS['models']}/hypernetworks",
        '--localizations-dir', PATHS['localizations'],
        '--codeformer-models-path', f"{PATHS['models']}/Codeformer",
        '--gfpgan-models-path', f"{PATHS['models']}/GFPGAN",
        '--esrgan-models-path', f"{PATHS['models']}/ESRGAN",
        '--bsrgan-models-path', f"{PATHS['models']}/BSRGAN",
        '--realesrgan-models-path', f"{PATHS['models']}/RealESRGAN",
        '--scunet-models-path', f"{PATHS['models']}/ScuNET",
        '--swinir-models-path', f"{PATHS['models']}/SwinIR",
        '--ldsr-models-path', f"{PATHS['models']}/LDSR",

        '--styles-file', f"{PATHS['styles_file']}",
        '--ui-config-file', f"{PATHS['ui_config_file']}",
        '--ui-settings-file', f"{PATHS['ui_settings_file']}",
    ]

    cmd_args = [ '--skip-torch-cuda-test' ]

    if IN_COLAB:
        args.append('--lowram')

        # xformers
        if USE_XFORMERS:
            log('xformers 를 사용합니다')

            if has_python_package('xformers'):
                cmd_args.append('--xformers')

            elif IN_COLAB:
                log('xformers 패키지가 존재하지 않습니다, 미리 컴파일된 파일로부터 xformers 패키지를 가져옵니다')
                download('https://github.com/toriato/easy-stable-diffusion/raw/prebuilt-xformers/cu113/xformers-0.0.14.dev0-cp37-cp37m-linux_x86_64.whl')
                execute(
                    ['pip', 'install', 'xformers-0.0.14.dev0-cp37-cp37m-linux_x86_64.whl'],
                    summary='xformers 패키지를 설치합니다'
                )
                cmd_args.append('--xformers')

            else:
                # TODO: 패키지 빌드
                log('xformers 패키지가 존재하지 않습니다, --xformers 인자를 사용하지 않습니다')

        # deepdanbooru
        if USE_DEEPDANBOORU:
            log('deepbooru 를 사용합니다')
            cmd_args.append('--deepdanbooru')

        # gradio
        if USE_GRADIO_TUNNEL:
            log('Gradio 터널을 사용합니다')
            args.append('--share')

        # gradio 인증
        if GRADIO_USERNAME != '':
            # 다계정이 아니고 비밀번호가 없다면 무작위로 만들기
            if GRADIO_PASSWORD == '' and ';' not in GRADIO_USERNAME:
                from secrets import token_urlsafe
                GRADIO_PASSWORD = token_urlsafe(8)
                GRADIO_PASSWORD_GENERATED = True

            args += [
                f'--gradio-auth',
                GRADIO_USERNAME + ('' if GRADIO_PASSWORD == '' else ':' + GRADIO_PASSWORD)
            ]

        # ngrok
        if NGROK_API_TOKEN != '':
            log('ngrok 터널을 사용합니다')
            args += ['--ngrok', NGROK_API_TOKEN]

            if has_python_package('pyngrok') is None:
                log('ngrok 사용에 필요한 패키지가 존재하지 않습니다, 설치를 시작합니다')
                execute(['pip', 'install', 'pyngrok'])

        # 추가 인자
        # TODO: 받은 문자열을 리스트로 안나누고 그대로 사용할 수 있는지?
        if ADDITIONAL_ARGS != '':
            args.append(ADDITIONAL_ARGS)

    start_webui(args, env={'COMMANDLINE_ARGS': ' '.join(cmd_args)})

# ^c 종료 무시하기
except KeyboardInterrupt:
    pass

except:
    # 로그 위젯이 없다면 평범하게 오류 처리하기
    if not LOG_WIDGET:
        raise

    log_trace()
