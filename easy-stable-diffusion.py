# @title
# fmt: off
import os
import sys
import platform
import re
import glob
import json
import requests
import torch
from typing import Union, Callable, Tuple, List
from subprocess import Popen, PIPE, STDOUT
from distutils.spawn import find_executable
from importlib.util import find_spec
from pathlib import Path
from os import makedirs
from shutil import copy, copytree, move, rmtree
from datetime import datetime
from IPython.display import display
from ipywidgets import widgets

def format_styles(styles: dict) -> str:
    return ';'.join(map(lambda kv: ':'.join(kv), styles.items()))

html_logger = widgets.HTML()
html_logger.blocks = []

def append_log_block(summary: str='', lines: List[str]=None,
                        summary_styles: dict=None, line_styles: dict=None,
                        fold=False, max_render_lines=5) -> int:
    block = {**locals()}

    if block['lines'] is None: block['lines'] = []
    if block['summary_styles'] is None: block['summary_styles'] = {}
    if block['line_styles'] is None: block['line_styles'] = {}

    html_logger.blocks.append(block)
    return len(html_logger.blocks) - 1

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

    for block in html_logger.blocks:
        summary_styles = {
            'display': 'inline-block',
            'cursor': 'pointer' if len(block['lines']) > 0 else 'inherit',
            **block['summary_styles']
        }
        line_styles = {
            'display': 'inline-block',
            **block['line_styles']
        }

        html += '<details " ' + ('' if block['fold'] else 'open=""') + '>'
        html += f'<summary style="{format_styles(summary_styles)}">{block["summary"]}</summary>'

        if len(block['lines']) > 0:
            html += f'<div style="{line_styles}">'
            html += ''.join(block['lines'][-block['max_render_lines']:])
            html += '</div>'

        html += '</details>'

    html += '</div>'

    html_logger.value = html

def log(msg: str, styles={}, newline=True, block_index: int=None) -> None:
    # 기록할 내용이 ngrok API 키와 일치한다면 숨기기
    # TODO: 더 나은 문자열 검사, 원치 않은 내용이 가려질 수도 있음
    if NGROK_API_KEY != '':
        msg = msg.replace(NGROK_API_KEY, '**REDACTED**')

    if newline:
        msg += '\n'

    # 파일 기록 추가
    if LOG_FILE:
        if block_index and msg.endswith('\n'):
            LOG_FILE.write('\t')
        LOG_FILE.write(msg)
        LOG_FILE.flush()

    if block_index is None:
        block_index = append_log_block(summary=msg, summary_styles=styles)
    else:
        html_logger.blocks[block_index]['lines'].append(msg)

    render_log()


# ==============================
# 서브 프로세스
# ==============================
running_subprocess = None

def execute(args: Union[str, List[str]], parser: Callable=None,
            logging=True, throw=True, **kwargs) -> Tuple[str, Popen]:
    global running_subprocess

    # 이미 서브 프로세스가 존재한다면 예외 처리하기
    if running_subprocess:
        raise Exception('이미 다른 하위 프로세스가 실행되고 있습니다')

    block_index = append_log_block(
        summary=f"=> {args if isinstance(args, str) else ' '.join(args)}\n",
        summary_styles={
            'color': 'yellow',
        },
        line_styles={
            'padding-left': '1.5em',
            'color': 'gray',
        }
    )

    # 서브 프로세스 만들기
    running_subprocess = Popen(
        args,
        stdout=PIPE,
        stderr=STDOUT,
        encoding='utf-8',
        **kwargs,
    )
    running_subprocess.output = ''
    running_subprocess.block_index = block_index

    # 프로세스 출력 위젯에 리다이렉션하기
    while running_subprocess.poll() is None:
        # 출력이 비어있다면 넘어가기
        out = running_subprocess.stdout.readline()
        if not out:
            continue

        # 프로세스 출력 버퍼에 추가하기
        running_subprocess.output += out

        # 파서가 없거나 또는 파서 실행 후 반환 값이 거짓이라면 로깅하기
        if (not parser or not parser(out)) and logging:
            log(out, block_index=block_index, newline=False)

    # 변수 정리하기
    output = running_subprocess.output
    returncode = running_subprocess.poll()
    running_subprocess = None

    # html_logger.blocks[block_index]['summary'] += f' -> {returncode}'

    # 반환 코드가 정상이 아니라면
    if returncode == 0:
        html_logger.blocks[block_index]['summary_styles']['color'] = 'green'
        html_logger.blocks[block_index]['fold'] = True

    else:
        html_logger.blocks[block_index]['summary_styles']['color'] = 'red'
        html_logger.blocks[block_index]['max_render_lines'] = 0

        if throw:
            raise Exception(f'프로세스가 {returncode} 코드를 반환했습니다')

    return output, returncode


# ==============================
# 작업 경로
# ==============================
path_to = {}

def update_path_to(path_to_workspace: str) -> None:
    global LOG_FILE

    path_to['workspace'] = path_to_workspace
    path_to['outputs'] = f"{path_to['workspace']}/outputs"
    path_to['models'] = f"{path_to['workspace']}/models"
    path_to['embeddings'] = f"{path_to['workspace']}/embeddings"
    path_to['scripts'] = f"{path_to['workspace']}/scripts"
    path_to['logs'] = f"{path_to['workspace']}/logs"
    path_to['styles_file'] = f"{path_to['workspace']}/styles.csv"
    path_to['ui_config_file'] = f"{path_to['workspace']}/ui-config.json"
    path_to['ui_settings_file'] = f"{path_to['workspace']}/config.json"

    makedirs(path_to['workspace'], exist_ok=True)
    makedirs(path_to['embeddings'], exist_ok=True)
    makedirs(path_to['scripts'], exist_ok=True)
    makedirs(path_to['logs'], exist_ok=True)

    log_path = os.path.join(path_to['logs'], datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M-%S.log'))

    # 기존 로그 파일이 존재한다면 옮기기
    if LOG_FILE:
        LOG_FILE.close()
        move(LOG_FILE.name, log_path)

    LOG_FILE = open(log_path, 'a')


# ==============================
# 사용자 설정
# ==============================
CHECKPOINTS = {
    # 'Standard Model 1.4': {
    #   'files': [{'url': 'https://public.vmm.pw/aeon/models/sd-v1-4.ckpt'}]
    # },

    # NAI leaks
    'NAI - animefull-final-pruned': {
        'files': [
            {
                # 'url': 'https://anonfiles.com/n6h3Q0Bdyf',
                'url': 'https://cloudflare-ipfs.com/ipfs/bafybeicpamreyp2bsocyk3hpxr7ixb2g2rnrequub3j2ahrkdxbvfbvjc4/model.ckpt',
                'target': 'nai-animefull-final-pruned.ckpt',
            },
            {
                # 'url': 'https://anonfiles.com/66c1QcB7y6',
                'url': 'https://cloudflare-ipfs.com/ipfs/bafybeiccldswdd3wvg57jhclcq53lvsc6gizasiblwayvhlv6eq4wow7wu/animevae.pt',
                'target': 'nai-animefull-final-pruned.vae.pt'
            },
            {
                'url': 'https://gist.githubusercontent.com/toriato/ae1f587f4d1e9ee5d0e910c627277930/raw/6019f8782875497f6e5b3e537e30a75df5b64812/animefull-final-pruned.yaml',
                'target': 'nai-animefull-final-pruned.yaml'
            }
        ]
    },
    'NAI - animefull-latest': {
        'files': [
            {
                'url': 'https://anonfiles.com/8fm7QdB1y9',
                'target': 'nai-animefull-latest.ckpt'
            },
            {
                'url': 'https://anonfiles.com/66c1QcB7y6',
                'target': 'nai-animefull-latest.vae.pt'
            },
            {
                'url': 'https://gist.githubusercontent.com/toriato/ae1f587f4d1e9ee5d0e910c627277930/raw/6019f8782875497f6e5b3e537e30a75df5b64812/animefull-latest.yaml',
                'target': 'nai-animefull-latest.yaml'
            }
        ]
    },
    # 'NAI - animefull-final-pruned': {
    #     'files': [
    #         {
    #             'url': 'https://drive.google.com/uc?id=1N6Zg4nJYnz7nN-vF8KExw8UbOwHddh12',
    #             'target': 'nai-animefull-final-pruned.ckpt'
    #         },
    #         {
    #             'url': 'https://drive.google.com/uc?id=1MnVdsJAFeIbhKn_-wvahuOUCOAhLjptb',
    #             'target': 'nai-animefull-final-pruned.vae.pt'
    #         },
    #         {
    #             'url': 'https://gist.githubusercontent.com/toriato/ae1f587f4d1e9ee5d0e910c627277930/raw/6019f8782875497f6e5b3e537e30a75df5b64812/animefull-final-pruned.yaml',
    #             'target': 'nai-animefull-final-pruned.yaml'
    #         }
    #     ]
    # },
    # 'NAI - animefull-latest': {
    #     'files': [
    #         {
    #             'url': 'https://drive.google.com/uc?id=173P4agYFiaYP1UDvPIocr3GnIzToabjh',
    #             'target': 'nai-animefull-latest.ckpt'
    #         },
    #         {
    #             'url': 'https://drive.google.com/uc?id=1MnVdsJAFeIbhKn_-wvahuOUCOAhLjptb',
    #             'target': 'nai-animefull-latest.vae.pt'
    #         },
    #         {
    #             'url': 'https://gist.githubusercontent.com/toriato/ae1f587f4d1e9ee5d0e910c627277930/raw/6019f8782875497f6e5b3e537e30a75df5b64812/animefull-latest.yaml',
    #             'target': 'nai-animefull-latest.yaml'
    #         }
    #     ]
    # },
    # 'NAI - animesfw-final-pruned': {
    #     'files': [
    #         {
    #             'url': 'https://drive.google.com/uc?id=1aNaA5utAHuj0vISnxQvC2bHz2AyZbkyv',
    #             'target': 'nai-animesfw-final-pruned.ckpt'
    #         },
    #         {
    #             'url': 'https://drive.google.com/uc?id=1MnVdsJAFeIbhKn_-wvahuOUCOAhLjptb',
    #             'target': 'nai-animesfw-final-pruned.vae.pt'
    #         },
    #         {
    #             'url': 'https://gist.githubusercontent.com/toriato/ae1f587f4d1e9ee5d0e910c627277930/raw/6019f8782875497f6e5b3e537e30a75df5b64812/animesfw-final-pruned.yaml',
    #             'target': 'nai-animesfw-final-pruned.yaml'
    #         }
    #     ]
    # },
    # 'NAI - animesfw-latest': {
    #     'files': [
    #         {
    #             'url': 'https://drive.google.com/uc?id=1N5Nla5e2SowFHYH3nuogV2IvigET7a4k',
    #             'target': 'nai-animesfw-final-pruned.ckpt'
    #         },
    #         {
    #             'url': 'https://drive.google.com/uc?id=1MnVdsJAFeIbhKn_-wvahuOUCOAhLjptb',
    #             'target': 'nai-animesfw-final-pruned.vae.pt'
    #         },
    #         {
    #             'url': 'https://gist.githubusercontent.com/toriato/ae1f587f4d1e9ee5d0e910c627277930/raw/6019f8782875497f6e5b3e537e30a75df5b64812/animesfw-latest.yaml',
    #             'target': 'nai-animesfw-final-pruned.yaml'
    #         }
    #     ]
    # },

    # Waifu stuffs
    # 'Waifu Diffusion 1.2': {
    #   'files': [{'url': 'https://public.vmm.pw/aeon/models/wd-v1-2-full-ema-pruned.ckpt'}]
    # },
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

    # Kinky c:
    # 'gg1342_testrun1': {
    #   'files': [{'url': 'https://public.vmm.pw/aeon/models/gg1342_testrun1_pruned.ckpt'}]
    # },
    # 'Hentai Diffusion RD1412': {
    #   'files': [{
    #     'url': 'https://public.vmm.pw/aeon/models/RD1412-pruned-fp16.ckpt',
    #     'args': ['-o', 'hentai_diffusion-rd1412-pruned-fp32.ckpt']
    #   }]
    # },
    # 'Bare Feet / Full Body b4_t16_noadd': {
    #   'files': [{'url': 'https://public.vmm.pw/aeon/models/bf_fb_v3_t4_b16_noadd-ema-pruned-fp16.ckpt'}]
    # },
    # 'Lewd Diffusion 70k (epoch 2)': {
    #   'files': [{'url': 'https://public.vmm.pw/aeon/models/LD-70k-2e-pruned.ckpt'}]
    # },

    # More kinky c:<
    # 'Yiffy (epoch 18)': {
    #   'files': [{'url': 'https://public.vmm.pw/aeon/models/yiffy-e18.ckpt'}]
    # },
    'Furry (epoch 4)': {
        'files': [{'url': 'https://iwiftp.yerf.org/Furry/Software/Stable%20Diffusion%20Furry%20Finetune%20Models/Finetune%20models/furry_epoch4.ckpt'}]
    },
    'Zack3D Kinky v1': {
        'files': [{'url': 'https://iwiftp.yerf.org/Furry/Software/Stable%20Diffusion%20Furry%20Finetune%20Models/Finetune%20models/Zack3D_Kinky-v1.ckpt'}]
    },
    # 'R34 (epoch 1)': {
    #   'files': [{'url': 'https://public.vmm.pw/aeon/models/r34_e1.ckpt'}]
    # },
    # 'Pony Diffusion': {
    #  'files': [{ 'url': 'https://public.vmm.pw/aeon/models/pony_sfw_80k_safe_and_suggestive_500rating_plus-pruned.ckpt'}]
    # },
    'Pokemon': {
        'files': [{
            'url': 'https://huggingface.co/justinpinkney/pokemon-stable-diffusion/resolve/main/ema-only-epoch%3D000142.ckpt',
            'args': ['-o', 'pokemon-ema-pruned.ckpt']
        }]
    },

    # Others...
    'Dreambooth - Hiten': {
        'files': [{'url': 'https://huggingface.co/BumblingOrange/Hiten/resolve/main/Hiten%20girl_anime_8k_wallpaper_4k.ckpt'}]
    },
}

# @markdown ### <font color="orange">***체크포인트 모델 선택***</font>
# @markdown - [모델 별 설명 및 다운로드 주소](https://rentry.org/sdmodels)
CHECKPOINT = 'NAI - animefull-final-pruned' # @param ['NAI - animefull-final-pruned', 'NAI - animefull-latest', 'Waifu Diffusion 1.3', 'Trinart Stable Diffusion v2 60,000 Steps', 'Trinart Stable Diffusion v2 95,000 Steps', 'Trinart Stable Diffusion v2 115,000 Steps', 'Furry (epoch 4)', 'Zack3D Kinky v1', 'Pokemon', 'Dreambooth - Hiten'] {allow-input: true}

# @markdown ### <font color="orange">***구글 드라이브 동기화를 사용할지?***</font>
USE_GOOGLE_DRIVE = True  # @param {type:"boolean"}

# @markdown ### <font color="orange">***구글 드라이브 작업 디렉터리 경로***</font>
# @markdown 임베딩, 모델, 결과, 설정 등 영구적으로 보관될 파일이 저장될 디렉터리의 경로
PATH_TO_GOOGLE_DRIVE = 'SD' # @param {type:"string"}

# @markdown ### <font color="orange">***xformers 를 사용할지?***</font>
# @markdown - <font color="green">장점</font>: 켜두면 10-15% 정도의 성능 향상을 *보일 수도 있음*
# @markdown - <font color="red">단점</font>: 켜두면 코랩을 제외한 환경에서 패키지를 새로 컴파일해야함
USE_XFORMERS = True  # @param {type:"boolean"}

# @markdown ### <font color="orange">***DeepDanbooru 를 사용할지?***</font>
# @markdown IMG2IMG 에 올린 이미지의 프롬프트를 단부루 태그 형태로 예측해주는 기능
# @markdown - <font color="green">장점</font>: 켜두면 10-15% 정도의 성능 향상을 *보일 수도 있음*
# @markdown - <font color="red">단점</font>: 켜두면 준비 시간이 조금 느려질 수 있음
USE_DEEPDANBOORU = True  # @param {type:"boolean"}

# @markdown ##### <font size="2" color="red">(선택)</font> <font color="orange">***Graido 인증 정보***</font>
# @markdown Gradio 접속 시 사용할 사용자, 비밀번호 정보
# @markdown <br>`GRADIO_USERNAME`에 `user1:pass1,user2,pass2` 형태를 입력하면 여러 사용자 등록 가능
GRADIO_USERNAME = '' # @param {type:"string"}
GRADIO_PASSWORD = '' # @param {type:"string"}

# @markdown ##### <font size="2" color="red">(선택)</font> <font color="orange">***ngrok API 키***</font>
# @markdown [API 키 만들기](https://dashboard.ngrok.com/get-started/your-authtoken)
NGROK_API_KEY = '' # @param {type:"string"}

# @markdown ##### <font size="2" color="red">(선택)</font> <font color="orange">***WebUI 레포지토리 주소***</font>
REPO_URL = 'https://github.com/AUTOMATIC1111/stable-diffusion-webui.git' # @param {type:"string"}

# @markdown ##### <font size="2" color="red">(선택)</font> <font color="orange">***WebUI 레포지토리 커밋 해시***</font>
REPO_COMMIT = '' # @param {type:"string"}

# 레포지토리에 적용할 풀 리퀘스트
REPO_PULL_REQUESTS = []

# 추가로 받을 스크립트
ADDITIONAL_SCRIPTS = [
    # 태그 자동 완성 유저스크립트
    # https://arca.live/b/aiart/60536925/272094058
    lambda: download(
        'https://greasyfork.org/scripts/452929-webui-%ED%83%9C%EA%B7%B8-%EC%9E%90%EB%8F%99%EC%99%84%EC%84%B1/code/WebUI%20%ED%83%9C%EA%B7%B8%20%EC%9E%90%EB%8F%99%EC%99%84%EC%84%B1.user.js',
        'repo/javascript'
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
    lambda: download(
        'https://raw.githubusercontent.com/jtkelm2/stable-diffusion-webui-1/master/scripts/wildcards.py',
        'repo/scripts'
    ),

    # txt2mask
    # https://github.com/ThereforeGames/txt2mask
    [
        lambda: rmtree('.tmp', ignore_errors=True),
        lambda: makedirs('.tmp', exist_ok=True),
        lambda: execute('curl -sSL https://github.com/ThereforeGames/txt2mask/tarball/master | tar xzvf - --strip-components=1 -C .tmp', shell=True),
        lambda: rmtree('repo/repositories/clipseg', ignore_errors=True),
        lambda: copytree('.tmp/repositories/clipseg', 'repo/repositories/clipseg'),
        lambda: copy('.tmp/scripts/txt2mask.py', 'repo/scripts'),
        lambda: rmtree('.tmp', ignore_errors=True),
    ],

    # Img2img Video
    # https://github.com/memes-forever/Stable-diffusion-webui-video
    [
        lambda: download(
            'https://raw.githubusercontent.com/memes-forever/Stable-diffusion-webui-video/main/videos.py',
            'repo/scripts'
        )
    ],

    # Seed Travel
    # https://github.com/yownas/seed_travel
    [
        lambda: execute('pip', 'install', 'moviepy') if find_spec('moviepy') is None else None,
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
        lambda: execute('pip', 'install', 'moviepy') if find_spec('moviepy') is None else None,
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
        lambda: execute('pip', 'install', 'moviepy') if find_spec('moviepy') is None else None,
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
        'repo/scritps'
    )
]

# @markdown ##### <font size="2" color="red">(선택)</font> <font color="orange">***WebUI 추가 인자***</font>
ADDITIONAL_ARGS = '' # @param {type:"string"}

# 로그 파일
LOG_FILE = None

# 현재 코랩 환경에서 구동 중인지?
IN_COLAB = find_spec('google') and find_spec('google.colab')

# ==============================
# 패키지 준비
# ==============================
def prepare_aria2() -> None:
    if find_executable('aria2c') is None:
        log('aria2c 명령어가 존재하지 않습니다, 설치를 시작합니다')
        execute(['sudo', 'apt', 'install', '-y', 'aria2'])

    # 설정 파일 만들기
    path_to_config = os.path.join(Path.home(), '.aria2', 'aria2.conf')
    if os.path.isfile(path_to_config):
        return

    log('aria2 설정 파일이 존재하지 않습니다, 추천 값으로 설정합니다')
    makedirs(os.path.dirname(path_to_config), exist_ok=True)
    with open(path_to_config, 'w') as f:
        f.write("""
summary-interval=10
allow-overwrite=true
always-resume=true
disk-cache=64M
continue=true
min-split-size=8M
max-concurrent-downloads=8
max-connection-per-server=8
max-overall-download-limit=0
max-download-limit=0
split=8
seed-time=0
""")


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
def download(url: str, target='', args=[]):
    if os.path.isdir(target) or target.endswith('/'):
        dirname = target
    else:
        dirname = os.path.dirname(target)
        basename = os.path.basename(target)

        if '-o' not in args and not os.path.isdir(target):
            args = ['-o', basename, *args]

    if dirname:
        makedirs(dirname, exist_ok=True)
        if '-d' not in args:
            args = ['-d', dirname, *args]

    if url.startswith('https://drive.google.com'):
        if find_spec('gdown') is None:
            execute(['pip', 'install', 'gdown'])

        execute(['gdown', '-O', target, url])
        return

    # anonfile CDN 주소 가져오기
    if url.startswith('https://anonfiles.com/'):
        matches = re.search('https://cdn-[^\"]+', requests.get(url).text)
        if not matches:
            raise Exception('anonfiles 에서 CDN 주소를 파싱하는데 실패했습니다')

        url = matches[0]

    prepare_aria2()
    execute(['aria2c', *args, url])

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
        target = os.path.join(f"{path_to['models']}/Stable-diffusion", file.get('target', ''))
        download(**{**file, 'target': target})

def has_checkpoint() -> bool:
    for p in Path(f"{path_to['models']}/Stable-diffusion").glob('**/*.ckpt'):
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
    # 모델 용량이 너무 커서 코랩 메모리 할당량을 초과하면 프로세스를 강제로 초기화됨
    # 이를 해결하기 위해선 모델 맵핑 위치를 VRAM으로 변경해줘야함
    # Thanks to https://gist.github.com/td2sk/e32a39344537fb3cd756ef4abdd3d371
    # TODO: 코랩에서만 발생하는 문제인지?
    log('모델 맵핑 위치를 변경합니다')
    execute([
        'sed',
        '-i',
        '''s/map_location="cpu"/map_location=torch.device("cuda")/g''',
        f"repo/modules/sd_models.py"
    ])

    # 기본 UI 설정 값 (ui-config.json)
    # 설정 파일 자체를 덮어씌우면 새로 추가된 키를 인식하지 못해서 코드 자체를 수정함
    # https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/shared.py
    path_to_shared = f"repo/modules/shared.py"
    if os.path.isfile(path_to_shared):
        log('설정 파일의 기본 값을 추천 값으로 변경합니다')

        configs = {
            # 결과 이미지 디렉터리
            'outdir_txt2img_samples': f"{path_to['outputs']}/txt2img-samples",
            'outdir_img2img_samples': f"{path_to['outputs']}/img2img-samples",
            'outdir_extras_samples': f"{path_to['outputs']}/extras-samples",
            'outdir_txt2img_grids': f"{path_to['outputs']}/txt2img-grids",
            'outdir_img2img_grids': f"{path_to['outputs']}/img2img-grids",

            # NAI 기본 설정(?)
            'eta_ancestral': 0.2,
            'CLIP_stop_at_last_layers': 2,
        }

        with open(path_to_shared, 'r+') as f:
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
    if not os.path.isfile(path_to['ui_config_file']):
        log('UI 설정 파일이 존재하지 않습니다, 추천 값으로 새로 생성합니다')

        with open(path_to['ui_config_file'], 'w') as f:
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
    for job in ADDITIONAL_SCRIPTS:
        if callable(job):
            job()
        elif isinstance(job, list):
            for child_job in job:
                child_job()

    # 사용자 스크립트 심볼릭 생성
    log('사용자 스크립트의 심볼릭 링크를 만듭니다')
    for path in os.listdir(path_to['scripts']):
        src = os.path.join(path_to['scripts'], path)
        dst = os.path.join('repo/scripts', os.path.basename(path))

        # 이미 파일이 존재한다면 기존 파일 삭제하기
        if os.path.lexists(dst):
            rmtree(dst) if os.path.isdir(dst) and not os.path.islink(dst) else os.remove(dst)

        # 심볼릭 링크 생성
        os.symlink(src, dst, target_is_directory=os.path.isdir(path))


def setup_webui() -> None:
    need_clone = True

    # 이미 디렉터리가 존재한다면 정상적인 레포인지 확인하기
    if os.path.isdir('repo'):
        try:
            log('레포지토리를 업데이트 합니다')

            # 사용자 파일만 남겨두고 레포지토리 초기화하기
            # https://stackoverflow.com/a/12096327
            execute(['git', 'add', '--ignore-errors', '-f', 'repositories'], cwd='repo')
            execute(['git', 'checkout', '.'], cwd='repo')
            execute(['git', 'reset', '--hard'], cwd='repo')
            execute(['git', 'pull'], cwd='repo')

            need_clone = False

        except:
            log('레포지토리가 잘못됐습니다, 디렉터리를 제거합니다')

    if need_clone:
        log('레포지토리를 가져옵니다')
        rmtree('repo', ignore_errors=True)
        execute(['git', 'clone', REPO_URL, 'repo'])

    # 특정 커밋이 지정됐다면 체크아웃하기
    if REPO_COMMIT != '':
        log(f'레포지토리를 {REPO_COMMIT} 커밋으로 되돌립니다')
        execute(['git', 'checkout', REPO_COMMIT])

    patch_webui_repository()

    # 코랩에선 필요 없으나 다른 환경에선 높은 확률로 설치 필요한 패키지들
    if not IN_COLAB:
        execute(['sudo', 'apt', 'install', '-y', 'build-essential', 'libgl1', 'libglib2.0-0'])

def parse_webui_output(out: str) -> bool:
    # 하위 스크립트 실행 중 오류가 발생하면 전체 기록 표시하기
    # TODO: 더 나은 오류 핸들링, 잘못된 내용으로 트리거 될 수 있음
    if 'Traceback (most recent call last):' in out:
        html_logger.blocks[running_subprocess.block_index]['max_render_lines'] = 0

    # 외부 주소 출력되면 성공적으로 실행한 것으로 판단
    matches = re.search('https?://(\d+\.gradio\.app|[0-9a-f-]+\.ngrok\.io)', out)
    if matches:
        log(f'성공적으로 웹UI를 실행했습니다, 아래 주소에 접속해주세요!\n{matches[0]}',
            styles={
                'background-color': 'green',
                'font-weight': 'bold',
                'font-size': '1.5em',
                'line-height': '1em',
                'color': 'black'
            })

def start_webui(args: List[str]=[], env={}) -> None:
    global running_subprocess

    if running_subprocess is not None:
        if 'launch.py' in running_subprocess.args:
            log('이미 실행 중인 웹UI를 종료하고 다시 시작합니다')
            running_subprocess.kill()
            running_subprocess = None

        raise ('이미 다른 프로세스가 실행 중입니다, 잠시 후에 실행해주세요')

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
# 보고서
# ==============================
def generate_report() -> str:
    import traceback

    def format_list(value):
        if isinstance(value, dict):
            return '\n'.join(map(lambda kv: f'{kv[0]}: {kv[1]}', value.items()))
        else:
            return '\n'.join(value)

    # 스택 가져오기
    ex_type, ex_value, ex_traceback = sys.exc_info()
    traces = map(lambda v: f'{v[0]}#{v[1]}\n\t{v[2]}\n\t{v[3]}', traceback.extract_tb(ex_traceback))

    # 로그 가져오기
    logs = ''
    if LOG_FILE:
        with open(LOG_FILE.name) as file:
            logs = file.read()

    payload = f"""
{logs}
# {ex_type.__name__}: {ex_value}
{format_list(traces)}

# options
CHECKPOINT: {CHECKPOINT}
USE_GOOGLE_DRIVE: {USE_GOOGLE_DRIVE}
PATH_TO_GOOGLE_DRIVE: {PATH_TO_GOOGLE_DRIVE}
USE_DEEPDANBOORU: {USE_DEEPDANBOORU}
GRADIO_USERNAME: {GRADIO_USERNAME != ''}
GRADIO_PASSWORD: {GRADIO_PASSWORD != ''}
NGROK_API_KEY: {NGROK_API_KEY != ''}
REPO_URL: {REPO_URL}
ADDITIONAL_ARGS: {ADDITIONAL_ARGS}

# paths
{format_list(path_to)}

# models
{format_list(glob.glob(f"{path_to['models']}/**/*"))}
"""

    res = requests.post('https://hastebin.com/documents',
                        data=payload.encode('utf-8'))

    return f"https://hastebin.com/{json.loads(res.text)['key']}"


# ==============================
# 자 드게제~
# ==============================
try:
    # 코랩 폼 입력 란을 생성을 위한 코드
    # log(', '.join(map(lambda s:f"'{s}'", CHECKPOINTS.keys())))

    # 인터페이스 출력
    btn_download_checkpoint = widgets.Button(description='체크포인트 받기')
    btn_download_checkpoint.on_click(
        lambda _: download_checkpoint(CHECKPOINT)
    )

    display(
        widgets.VBox([
            btn_download_checkpoint,
            html_logger
        ])
    )

    # 기본 작업 경로 설정
    update_path_to(os.path.abspath(os.curdir))

    log(platform.platform())
    log(f'Python {platform.python_version()}')
    log('')

    if IN_COLAB:
        log('현재 코랩을 사용하고 있습니다')
        makedirs('/usr/local/content', exist_ok=True)
        os.chdir('/usr/local/content')

        assert torch.cuda.is_available(), 'GPU 가 없습니다, 런타임 유형이 잘못됐거나 GPU 할당량이 초과된 것 같습니다'

        # 구글 드라이브 마운팅 시도
        if USE_GOOGLE_DRIVE:
            mount_google_drive()

    # 체크포인트가 없을 시 다운로드
    if not has_checkpoint():
        download_checkpoint(CHECKPOINT)

    # WebUI 가져오기
    setup_webui()

    # WebUI 실행
    args = [
        # 동적 경로들
        f"--ckpt-dir={path_to['models']}/Stable-diffusion",
        f"--embeddings-dir={path_to['embeddings']}",
        f"--hypernetwork-dir={path_to['models']}/hypernetworks",
        f"--codeformer-models-path={path_to['models']}/Codeformer",
        f"--gfpgan-models-path={path_to['models']}/GFPGAN",
        f"--esrgan-models-path={path_to['models']}/ESRGAN",
        f"--bsrgan-models-path={path_to['models']}/BSRGAN",
        f"--realesrgan-models-path={path_to['models']}/RealESRGAN",
        f"--scunet-models-path={path_to['models']}/ScuNET",
        f"--swinir-models-path={path_to['models']}/SwinIR",
        f"--ldsr-models-path={path_to['models']}/LDSR",

        f"--styles-file={path_to['styles_file']}",
        f"--ui-config-file={path_to['ui_config_file']}",
        f"--ui-settings-file={path_to['ui_settings_file']}",
    ]

    cmd_args = [ '--skip-torch-cuda-test' ]

    if USE_XFORMERS:
        if IN_COLAB and find_spec('xformers') is None:
            log('xformers 패키지가 존재하지 않습니다, 미리 컴파일된 파일로부터 설치를 시작합니다')
            download('https://github.com/toriato/easy-stable-diffusion/releases/download/xformers/xformers-0.0.14.dev0-cp37-cp37m-linux_x86_64.whl')
            execute(['pip', 'install', 'xformers-0.0.14.dev0-cp37-cp37m-linux_x86_64.whl'])

        cmd_args.append('--xformers')

    if USE_DEEPDANBOORU:
        cmd_args.append('--deepdanbooru')

    if NGROK_API_KEY == '':
        log('Gradio 터널을 사용합니다')
        args += ['--share', '--gradio-debug']

        # Gradio 인증 정보
        if GRADIO_USERNAME != '':
            args.append('--gradio-auth=' + GRADIO_USERNAME + ('' if GRADIO_PASSWORD == '' else ':' + GRADIO_PASSWORD))
    else:
        log('ngrok 터널을 사용합니다')
        args.append(f'--ngrok={NGROK_API_KEY}')

        if find_spec('pyngrok') is None:
            log('ngrok 사용에 필요한 패키지가 존재하지 않습니다, 설치를 시작합니다')
            execute(['pip', 'install', 'pyngrok'])

    if ADDITIONAL_ARGS != '':
        args.append(ADDITIONAL_ARGS)

    start_webui(
        args, 
        env={
            'COMMANDLINE_ARGS': ' '.join(cmd_args)
        }
    )

# ^c 종료 무시하기
except KeyboardInterrupt:
    pass

# 오류 발생하면 보고서 생성하고 표시하기
except:
    _, ex_value, _ = sys.exc_info()
    report_url = generate_report()

    log(f'{ex_value}\n오류가 발생했습니다, 아래 주소를 복사해 보고해주세요!\n{report_url}',
        styles={
            'background-color': 'red',
            'font-weight': 'bold',
            'font-size': '1.5em',
            'line-height': '1em',
            'color': 'black'
        })
