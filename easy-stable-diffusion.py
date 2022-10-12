# @title
# fmt: off
import os
import sys
import platform
import re
import glob
import json
import requests
from typing import Union, Callable, Tuple, List
from subprocess import Popen, PIPE, STDOUT
from importlib.util import find_spec
from pathlib import Path
from shutil import rmtree
from IPython.display import display
from ipywidgets import widgets

def format_styles(styles: dict) -> str:
    return ';'.join(map(lambda kv: ':'.join(kv), styles.items()))

html_dialog = widgets.HTML()
html_dialog_presets = {
    'default': {
        'display': 'inline-block',
        'padding': '.5em',
        'background-color': 'black',
        'font-size': '1.25em',
        'line-height': '1em',
        'color': 'white',
    },
    'error': {
        'border-left': '6px solid red'
    }
}

html_logger_styles = {
    'overflow-x': 'auto',
    'max-width': '700px',
    'padding': '1em',
    'background-color': 'black',
    'white-space': 'pre-wrap',
    'font-family': 'monospace',
    'font-size': '1em',
    'line-height': '1em',
    'color': 'white'
}
html_logger = widgets.HTML(
    value=f'<div id="logger" style="{format_styles(html_logger_styles)}">')
html_logger.raw = ''


def dialog(msg, preset:str=None, styles=html_dialog_presets['default']) -> None:
    if preset and preset in html_dialog_presets:
        styles = {
            **html_dialog_presets['default'],
            **html_dialog_presets[preset],
            **styles
        }

    html_dialog.value = f'<div style="{format_styles(styles)}">{msg}</div>'


def log(msg, newline=True, styles={}, bold=False) -> None:
    if bold:
        styles['font-weight'] = 'bold'

    if newline:
        msg += '\n'

    html_logger.raw += msg
    html_logger.value += f'<span style="{format_styles(styles)}">{msg}</span>'


# ==============================
# 서브 프로세스
# ==============================
running_subprocess = None


def execute(args: Union[str, List[str]], parser: Callable=None,
            logging=True, throw=True, **kwargs) -> Tuple[str, Popen]:
    global running_subprocess

    # 이미 서브 프로세스가 존재한다면 예외 처리하기
    if running_subprocess:
        raise Exception('하위 프로세스가 실행되고 있습니다')

    if isinstance(args, str):
        log(f'=> {args}', styles={'color':'yellow'})
    else:
        log(f"=> {' '.join(args)}", styles={'color':'yellow'})

    html_logger.value += '<div style="padding-left:1em">'

    # 서브 프로세스 만들기
    running_subprocess = Popen(
        args,
        stdout=PIPE,
        stderr=STDOUT,
        encoding='utf-8',
        **kwargs,
    )

    running_subprocess.output = ''

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
            log(out, newline=False, styles={'color': '#AAA'})

    # 변수 정리하기
    output = running_subprocess.output
    returncode = running_subprocess.poll()
    running_subprocess = None

    # 명령어 실행에 실패했다면 빨간 글씨로 표시하기
    styles = {'color': 'green'}
    if returncode != 0:
        styles['color'] = 'red'

    html_logger.value += '</div>'
    log(f"=> {returncode}", styles=styles)

    # 반환 코드가 정상이 아니라면 예외 발생하기
    if returncode != 0 and throw:
        raise Exception(f'프로세스가 {returncode} 코드를 반환했습니다')

    return output, returncode


# ==============================
# 작업 경로
# ==============================
path_to = {
    'repository': '/content/repository'
}

def update_path_to(path_to_workspace: str) -> None:
    log(f'작업 공간 경로를 "{path_to_workspace}" 으로 변경했습니다')

    path_to['workspace'] = path_to_workspace
    path_to['outputs'] = f"{path_to['workspace']}/outputs"
    path_to['models'] = f"{path_to['workspace']}/models"
    path_to['embeddings'] = f"{path_to['workspace']}/embeddings"
    path_to['styles_file'] = f"{path_to['workspace']}/styles.csv"
    path_to['ui_config_file'] = f"{path_to['workspace']}/ui-config.json"
    path_to['ui_settings_file'] = f"{path_to['workspace']}/config.json"

    os.makedirs(path_to['workspace'], exist_ok=True)
    os.makedirs(path_to['embeddings'], exist_ok=True)


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
                'url': 'https://anonfiles.com/n6h3Q0Bdyf',
                'args': ['-o', 'nai-animefull-final-pruned.ckpt']
            },
            {
                'url': 'https://anonfiles.com/66c1QcB7y6',
                'args': ['-o', 'nai-animefull-final-pruned.vae.pt']
            },
            {
                'url': 'https://gist.githubusercontent.com/toriato/ae1f587f4d1e9ee5d0e910c627277930/raw/6019f8782875497f6e5b3e537e30a75df5b64812/animefull-final-pruned.yaml',
                'args': ['-o', 'nai-animefull-final-pruned.yaml']
            }
        ]
    },
    'NAI - animefull-latest': {
        'files': [
            {
                'url': 'https://anonfiles.com/8fm7QdB1y9',
                'args': ['-o', 'nai-animefull-latest.ckpt']
            },
            {
                'url': 'https://anonfiles.com/66c1QcB7y6',
                'args': ['-o', 'nai-animefull-latest.vae.pt']
            },
            {
                'url': 'https://gist.githubusercontent.com/toriato/ae1f587f4d1e9ee5d0e910c627277930/raw/6019f8782875497f6e5b3e537e30a75df5b64812/animefull-latest.yaml',
                'args': ['-o', 'nai-animefull-latest.yaml']
            }
        ]
    },

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
PATH_TO_GOOGLE_DRIVE = 'SD'  # @param {type:"string"}

# @markdown ### <font color="orange">***xformers 를 사용할지?***</font>
# @markdown - <font color="green">장점</font>: 켜두면 10-15% 정도의 성능 향상을 *보일 수도 있음*
# @markdown - <font color="red">단점</font>: 켜두면 준비 시간이 아주 크게 늘어남 (1시간 이상)
USE_XFORMERS = False  # @param {type:"boolean"}

# @markdown ### <font color="orange">***DeepDanbooru 를 사용할지?***</font>
# @markdown IMG2IMG 에 올린 이미지의 프롬프트를 단부루 태그 형태로 예측해주는 기능
# @markdown - <font color="green">장점</font>: 켜두면 10-15% 정도의 성능 향상을 *보일 수도 있음*
# @markdown - <font color="red">단점</font>: 켜두면 준비 시간이 조금 느려질 수 있음
USE_DEEPDANBOORU = True  # @param {type:"boolean"}

# 현재 코랩 환경에서 구동 중인지?
IN_COLAB = find_spec('google.colab') is not None


# ==============================
# 패키지 준비
# ==============================
def prepare_aria2() -> None:
    log('aria2 패키지를 설치합니다')
    execute(
        ['sudo', 'apt', 'install', '-y', 'aria2'])

    # 설정 파일 만들기
    log('aria2 설정 파일을 만듭니다')
    os.makedirs(os.path.join(Path.home(), '.aria2'), exist_ok=True)
    with open(Path.joinpath(Path.home(), '.aria2', 'aria2.conf'), "w") as f:
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
    log('구글 드라이브 마운트를 시도합니다')

    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)

    # 전체 경로 업데이트
    update_path_to(os.path.join('/content/drive/MyDrive', PATH_TO_GOOGLE_DRIVE))


# ==============================
# 파일 다운로드
# ==============================
def download(url: str, args=[]):
    # anonfile CDN 주소 가져오기
    if url.startswith('https://anonfiles.com/'):
        matches = re.search('https://cdn-[^\"]+', requests.get(url).text)
        if not matches:
            raise Exception('anonfiles 에서 CDN 주소를 파싱하는데 실패했습니다')

        url = matches[0]

    # Aria2 로 모델 받기
    log(f"파일 다운로드를 시도합니다: {url}")
    execute(['aria2c', *args, url])
    log('파일을 성공적으로 받았습니다!')


def download_checkpoint(checkpoint: str) -> None:
    if checkpoint in CHECKPOINTS:
        checkpoint = CHECKPOINTS[checkpoint]
    else:
        # 미리 선언된 체크포인트가 아니라면 주소로써 사용하기
        checkpoint = {'files': [{'url': checkpoint}]}

    # Aria2 로 모델 받기
    # TODO: 토렌트 마그넷 주소 지원
    log(f"파일 {len(checkpoint['files'])}개를 받습니다")

    for f in checkpoint['files']:
        file = json.loads(json.dumps(f))

        if 'args' not in file:
            file['args'] = []

        # 모델 받을 기본 디렉터리 경로 잡아주기
        if '-d' not in file['args']:
            file['args'] = [
                '-d', f"{path_to['models']}/Stable-diffusion", *file['args']]

        download(**file)


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
        f"{path_to['repository']}/modules/sd_models.py"
    ])

    # 기본 UI 설정 값 (ui-config.json)
    # 설정 파일 자체를 덮어씌우면 새로 추가된 키를 인식하지 못해서 코드 자체를 수정함
    # https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/shared.py
    path_to_shared = f"{path_to['repository']}/modules/shared.py"
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
    # TODO: 이후 새 설정 값이 정상적으로 추가되는지 확인 필요함
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

def setup_webui() -> None:
    need_clone = True

    # 이미 디렉터리가 존재한다면 정상적인 레포인지 확인하기
    if os.path.isdir(path_to['repository']):
        try:
            log('WebUI 레포지토리를 풀(업데이트) 합니다')

            # 사용자 파일만 남겨두고 레포지토리 초기화하기
            # https://stackoverflow.com/a/12096327
            execute(['git', 'add', '--ignore-errors', '-f', 'repositories'], cwd=path_to['repository'])
            execute(['git', 'checkout', '.'], cwd=path_to['repository'])
            execute(['git', 'pull'], cwd=path_to['repository'])
            need_clone = False

        except:
            log('레포지토리가 잘못됐습니다, 디렉터리를 제거합니다')

    if need_clone:
        log('WebUI 레포지토리를 클론합니다')
        rmtree(path_to['repository'], ignore_errors=True)
        execute(['git', 'clone', 'https://github.com/AUTOMATIC1111/stable-diffusion-webui', path_to['repository']])

    patch_webui_repository()

    # 코랩에선 필요 없으나 다른 환경에선 높은 확률로 설치 필요한 패키지들
    if not IN_COLAB:
        execute(['sudo', 'apt', 'install', '-y', 'build-essential', 'libgl1', 'libglib2.0-0'])


def parse_webui_output(out: str) -> bool:
    matches = re.search('https://\d+\.gradio\.app', out)
    if matches:
        log('******************************************', styles={'color':'green'})
        log(f'성공적으로 웹UI를 실행했습니다, 아래 주소에 접속해주세요!\n{matches[0]}',
            styles={
                'background-color': 'green',
                'font-weight': 'bold',
                'font-size': '1.5em',
                'line-height': '1.5em',
                'color': 'black'
            })
        log('******************************************', styles={'color':'green'})

        dialog(f'''
        <p>성공적으로 웹UI를 실행했습니다!</p>
        <p><a target="_blank" href="{matches[0]}">{matches[0]}</a></p>
        ''')


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
        cwd=path_to['repository'],
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
    from distutils.spawn import find_executable

    ex_type, ex_value, ex_traceback = sys.exc_info()
    traces = map(lambda v: f'{v[0]}#{v[1]}\n\t{v[2]}\n\t{v[3]}', traceback.extract_tb(ex_traceback))

    packages, _ = execute(['pip', 'freeze'], logging=False, throw=False)

    def format_list(value):
        if isinstance(value, dict):
            return '\n'.join(map(lambda kv: f'{kv[0]}: {kv[1]}', value.items()))
        else:
            return '\n'.join(value)

    payload = f"""
{html_logger.raw}
# {ex_type.__name__}: {ex_value}
{format_list(traces)}

# options
CHECKPOINT: {CHECKPOINT}
USE_GOOGLE_DRIVE: {USE_GOOGLE_DRIVE}
PATH_TO_GOOGLE_DRIVE: {PATH_TO_GOOGLE_DRIVE}
USE_XFORMERS: {USE_XFORMERS}
USE_DEEPDANBOORU: {USE_DEEPDANBOORU}

# paths
{format_list(path_to)}

# models
{format_list(glob.glob(f"{path_to['models']}/**/*"))}

# python
{platform.platform()}
{sys.executable}
{packages}
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
            html_dialog,
            html_logger
        ])
    )

    # 기본 작업 경로 설정
    update_path_to(os.path.abspath(os.curdir))

    if IN_COLAB:
        log('코랩 환경이 감지됐습니다')

        import torch
        assert torch.cuda.is_available(), 'GPU 가 없습니다, 런타임 유형이 잘못됐거나 GPU 할당량이 초과된 것 같습니다'

        # 코랩 환경이라면 /content 디렉터리는 스토리지 속도가 느리기 때문에
        # /usr/local 속에서 구동할 필요가 있음
        log('레포지토리 디렉터리를 "/usr/local/repository" 로 변경합니다')
        path_to['repository'] = '/usr/local/repository'

        # 구글 드라이브 마운팅 시도
        if USE_GOOGLE_DRIVE:
            mount_google_drive()

    # 구동 필수 패키지 준비
    prepare_aria2()

    # 체크포인트가 없을 시 다운로드
    if not has_checkpoint():
        download_checkpoint(CHECKPOINT)

    # WebUI 가져오기
    setup_webui()

    # WebUI 실행
    args = [
        '--share',
        '--gradio-debug',

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
        cmd_args = [*cmd_args, '--xformers']

    if USE_DEEPDANBOORU:
        cmd_args = [*cmd_args, '--deepdanbooru']

    start_webui(args, env={'COMMANDLINE_ARGS': ' '.join(cmd_args)})

# ^c 종료 무시하기
except KeyboardInterrupt:
    pass

# 오류 발생하면 보고서 생성하고 표시하기
except:
    _, ex_value, _ = sys.exc_info()
    report_url = generate_report()

    log('******************************************', styles={'color':'red'})
    log(f'오류가 발생했습니다, 아래 주소를 복사해 보고해주세요!\n{report_url}',
        styles={
            'background-color': 'red',
            'font-weight': 'bold',
            'font-size': '1.5em',
            'line-height': '1.5em',
            'color': 'black'
        })
    log('******************************************', styles={'color':'red'})
    log(f'{ex_value}', styles={'color':'red'})

    dialog(
        f'''
        <p>오류가 발생했습니다, 아래 주소를 복사해 보고해주세요!</p>
        <p><strong>{generate_report()}</strong></p>
        ''',
        preset='error'
    )
