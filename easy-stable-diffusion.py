import io
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from distutils.spawn import find_executable
from importlib.util import find_spec
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import requests

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

#@markdown ##### <font color="orange">***인증 정보***</font>
#@markdown 접속 시 사용할 사용자 아이디와 비밀번호
#@markdown <br>`GRADIO_USERNAME` 입력 란에 `user1:pass1,user,pass2`처럼 입력하면 여러 사용자 추가 가능
#@markdown <br>`GRADIO_USERNAME` 입력 란을 <font color="red">비워두면</font> 인증 과정을 사용하지 않음
GRADIO_USERNAME = '' #@param {type:"string"}
GRADIO_PASSWORD = '' #@param {type:"string"}
OPTIONS['GRADIO_USERNAME'] = GRADIO_USERNAME
OPTIONS['GRADIO_PASSWORD'] = GRADIO_PASSWORD

#@markdown ##### <font color="orange">***터널링 서비스***</font>
TUNNEL = 'gradio' #@param ["none", "gradio", "cloudflared", "ngrok"]
TUNNEL_URL: Optional[str] = None
OPTIONS['TUNNEL'] = TUNNEL

#@markdown ##### <font color="orange">***ngrok API 키***</font>
#@markdown ngrok 터널에 사용할 API 토큰
#@markdown <br>[설정하는 방법은 여기를 클릭해 확인](https://arca.live/b/aiart/60683088), [API 토큰은 여기를 눌러 계정을 만든 뒤 얻을 수 있음](https://dashboard.ngrok.com/get-started/your-authtoken)
#@markdown <br>입력 란을 <font color="red">비워두면</font> ngrok 터널을 비활성화함
#@markdown - <font color="green">장점</font>: 접속이 빠른 편이고 타임아웃이 거의 발생하지 않음
#@markdown - <font color="red">**단점**</font>: 계정을 만들고 API 토큰을 직접 입력해줘야함
NGROK_API_TOKEN = '' #@param {type:"string"}
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
    """
    셀이 종료됐을 때 자동으로 런타임을 해제하도록 asyncio 스레드를 생성합니다
    """
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
        time.sleep(1)
        runtime.unassign()

    # 평범한 환경에선 비동기로 동작하여 바로 실행되나
    # 코랩 런타임에선 순차적으로 실행되기 때문에 현재 셀 종료 후 즉시 실행됨
    asyncio.create_task(unassign())


def setup_tunnels():
    global TUNNEL_URL

    tunnel = OPTIONS['TUNNEL']

    if tunnel == 'none':
        pass

    elif tunnel == 'gradio':
        if not has_python_package('gradio'):
            # https://fastapi.tiangolo.com/release-notes/#0910
            execute(['pip', 'install', 'gradio', 'fastapi==0.90.1'])

        import secrets

        from gradio.networking import setup_tunnel
        TUNNEL_URL = setup_tunnel('localhost', 7860, secrets.token_urlsafe(32))

    elif tunnel == 'cloudflared':
        if not has_python_package('pycloudflared'):
            execute(['pip', 'install', 'pycloudflared'])

        from pycloudflared import try_cloudflare
        TUNNEL_URL = try_cloudflare(port=7860).tunnel

    elif tunnel == 'ngrok':
        if not has_python_package('pyngrok'):
            execute(['pip', 'install', 'pyngrok'])

        auth = None
        token = OPTIONS['NGROK_API_TOKEN']

        if ':' in token:
            parts = token.split(':')
            auth = parts[1] + ':' + parts[-1]
            token = parts[0]

        from pyngrok import conf, exception, ngrok, process

        # 로컬 포트가 닫혀있으면 경고 메세지가 스팸마냥 출력되므로 오류만 표시되게 수정함
        process.ngrok_logger.setLevel('ERROR')

        try:
            tunnel = ngrok.connect(
                7860,
                pyngrok_config=conf.PyngrokConfig(
                    auth_token=token,
                    region='jp'
                ),
                auth=auth,
                bind_tls=True
            )
        except exception.PyngrokNgrokError:
            alert('ngrok 연결에 실패했습니다, 토큰을 확인해주세요!', True)
        else:
            assert isinstance(tunnel, ngrok.NgrokTunnel)
            TUNNEL_URL = tunnel.public_url

    else:
        raise ValueError(f'{tunnel} 에 대응하는 터널 서비스가 존재하지 않습니다')


def setup_environment():
    # 노트북 환경이라면 로그 표시를 위한 HTML 요소 만들기
    if IN_INTERACTIVE:
        try:
            from IPython.display import display
            from ipywidgets import widgets

            global LOG_WIDGET
            LOG_WIDGET = widgets.HTML()
            display(LOG_WIDGET)

        except ImportError:
            pass

    # 구글 드라이브 마운트하기
    if IN_COLAB and OPTIONS['USE_GOOGLE_DRIVE']:
        from google.colab import drive
        drive.mount('/content/drive')

        global WORKSPACE
        WORKSPACE = str(
            Path('drive', 'MyDrive', WORKSPACE).resolve()
        )

    # 로그 파일 만들기
    global LOG_FILE
    workspace = Path(WORKSPACE).resolve()
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

    if IN_COLAB:
        # 다른 Python 버전 설치
        if OPTIONS['PYTHON_EXECUTABLE'] and not find_executable(OPTIONS['PYTHON_EXECUTABLE']):
            execute(['apt', 'install', OPTIONS['PYTHON_EXECUTABLE']])
            execute(
                f"curl -sS https://bootstrap.pypa.io/get-pip.py | {OPTIONS['PYTHON_EXECUTABLE']}"
            )

        # 런타임이 정상적으로 초기화 됐는지 확인하기
        try:
            import torch
        except:
            alert('torch 패키지가 잘못됐습니다, 런타임을 다시 실행해주세요!', True)
        else:
            if not torch.cuda.is_available():
                alert('GPU 런타임이 아닙니다, 할당량이 초과 됐을 수도 있습니다!')

                OPTIONS['EXTRA_ARGS'] += [
                    '--skip-torch-cuda-test',
                    '--no-half',
                    '--opt-sub-quad-attention'
                ]

        # 코랩 tcmalloc 관련 이슈 우회
        # https://github.com/googlecolab/colabtools/issues/3412
        try:
            # 패키지가 이미 다운그레이드 됐는지 확인하기
            execute('dpkg -l libunwind8-dev', hide_summary=True)
        except subprocess.CalledProcessError:
            for url in (
                'http://launchpadlibrarian.net/367274644/libgoogle-perftools-dev_2.5-2.2ubuntu3_amd64.deb',
                'https://launchpad.net/ubuntu/+source/google-perftools/2.5-2.2ubuntu3/+build/14795286/+files/google-perftools_2.5-2.2ubuntu3_all.deb',
                'https://launchpad.net/ubuntu/+source/google-perftools/2.5-2.2ubuntu3/+build/14795286/+files/libtcmalloc-minimal4_2.5-2.2ubuntu3_amd64.deb',
                'https://launchpad.net/ubuntu/+source/google-perftools/2.5-2.2ubuntu3/+build/14795286/+files/libgoogle-perftools4_2.5-2.2ubuntu3_amd64.deb'
            ):
                download(url, ignore_aria2=True)
            execute('apt install -qq libunwind8-dev')
            execute('dpkg -i *.deb')
            execute('rm *.deb')

    # 외부 터널링 초기화
    setup_tunnels()

    # 체크포인트 모델이 존재하지 않는다면 기본 모델 받아오기
    if not has_checkpoint():
        for file in [
            {
                'url': 'https://huggingface.co/gsdf/Counterfeit-V2.5/resolve/main/Counterfeit-V2.5_fp16.safetensors',
                'target': str(workspace.joinpath('models/Stable-diffusion/Counterfeit-V2.5_fp16.safetensors')),
                'summary': '기본 체크포인트 파일을 받아옵니다'
            },
            {
                'url': 'https://huggingface.co/saltacc/wd-1-4-anime/resolve/main/VAE/kl-f8-anime2.ckpt',
                'target': str(workspace.joinpath(WORKSPACE, 'models/VAE/kl-f8-anime2.ckpt')),
                'summary': '기본 VAE 파일을 받아옵니다'
            }
        ]:
            download(**file)


# ==============================
# 로그
# ==============================


def format_styles(styles: dict) -> str:
    return ';'.join(map(lambda kv: ':'.join(kv), styles.items()))


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
        if len(LOG_BLOCKS[parent_index]['childs']) > 100:
            LOG_BLOCKS[parent_index]['childs'].pop(0)

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
            '\n'.join(traceback.format_tb(ex_traceback)),
            parent_index=parent_index
        )

    # 로그 파일이 없으면 보고하지 않기
    # TODO: 로그 파일이 존재하지 않을 수가 있나...?
    if not LOG_FILE:
        log('로그 파일이 존재하지 않습니다, 보고서를 만들지 않습니다')
        return


def alert(message: str, unassign=False):
    log(message)

    if IN_INTERACTIVE:
        from IPython.display import display
        from ipywidgets import widgets

        display(
            widgets.HTML(f'<script>alert({json.dumps(message)})</script>')
        )

    if IN_COLAB and unassign:
        from google.colab import runtime

        time.sleep(1)
        runtime.unassign()


# ==============================
# 서브 프로세스
# ==============================
def execute(
    args: Union[str, List[str]],
    parser: Optional[
        Callable[[str], None]
    ] = None,
    summary: Optional[str] = None,
    hide_summary=False,
    print_to_file=True,
    print_to_widget=True,
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
        max_childs=10)

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
            parser(line)

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
            # 현재 로그 텍스트 초록색으로 변경하고 프로세스 출력 숨기기
            LOG_BLOCKS[log_index]['styles']['color'] = 'green'
            LOG_BLOCKS[log_index]['max_childs'] = None
        else:
            # 현재 로그 텍스트 빨간색으로 변경하고 프로세스 출력 모두 표시하기
            LOG_BLOCKS[log_index]['styles']['color'] = 'red'
            LOG_BLOCKS[log_index]['max_childs'] = 0

        if hide_summary:
            # 현재 로그 블록 숨기기 (제거하기)
            del LOG_BLOCKS[log_index]

        # 로그 블록 렌더링
        render_log()

    # 오류 코드를 반환했다면
    if rc != 0:
        if isinstance(rc, signal.Signals):
            rc = rc.value

        raise subprocess.CalledProcessError(rc, args)

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
        ])

    return True if rc == 0 else False


# ==============================
# 파일 다운로드
# ==============================
def download(url: str, target: Optional[str] = None, ignore_aria2=False, **kwargs):
    if not target:
        # TODO: 경로 중 params 제거하기
        target = url.split('/')[-1]

    # 파일을 받을 디렉터리 만들기
    Path(target).parent.mkdir(0o777, True, True)

    # 빠른 다운로드를 위해 aria2 패키지 설치 시도하기
    if not ignore_aria2:
        if not find_executable('aria2c') and find_executable('apt'):
            execute(['apt', 'install', 'aria2'])

        if find_executable('aria2c'):
            p = Path(target)
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
                    '--dir', str(p.parent),
                    '--out', p.name,
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


def parse_webui_output(line: str) -> None:
    # 첫 시작에 한해서 웹 서버 열렸을 때 다이어로그 표시하기
    if line.startswith('Running on local URL:'):
        log(
            '\n'.join([
                '성공적으로 터널이 열렸습니다',
                f'<a target="_blank" href="{TUNNEL_URL}">{TUNNEL_URL}</a>',
            ]),
            LOG_WIDGET_STYLES['dialog_success']
        )
        return


def setup_webui() -> None:
    repo_dir = Path('repository')
    need_clone = True

    # 이미 디렉터리가 존재한다면 정상적인 레포인지 확인하기
    if repo_dir.is_dir():
        try:
            # 사용자 파일만 남겨두고 레포지토리 초기화하기
            # https://stackoverflow.com/a/12096327
            execute(
                'git stash && git pull',
                cwd=repo_dir
            )
        except subprocess.CalledProcessError:
            log('레포지토리가 잘못됐습니다, 디렉터리를 제거합니다')
        else:
            need_clone = False

    # 레포지토리 클론이 필요하다면 기존 디렉터리 지우고 클론하기
    if need_clone:
        shutil.rmtree(repo_dir, ignore_errors=True)
        execute(['git', 'clone', OPTIONS['REPO_URL'], str(repo_dir)])

    # 특정 커밋이 지정됐다면 체크아웃하기
    if OPTIONS['REPO_COMMIT']:
        execute(
            ['git', 'checkout', OPTIONS['REPO_COMMIT']],
            cwd=repo_dir
        )

    if IN_COLAB:
        patch_path = repo_dir.joinpath('scripts', 'patches.py')

        if not patch_path.exists():
            download(
                'https://raw.githubusercontent.com/toriato/easy-stable-diffusion/main/scripts/patches.py',
                str(patch_path),
                ignore_aria2=True)


def start_webui(args: List[str] = OPTIONS['ARGS']) -> None:
    workspace = Path(WORKSPACE).resolve()
    repository = Path('repository').resolve()

    # 기본 인자 만들기
    if len(args) < 1:
        args += ['--data-dir', str(workspace)]

        # xformers
        if OPTIONS['USE_XFORMERS']:
            try:
                import torch
            except ImportError:
                pass
            else:
                if torch.cuda.is_available():
                    args += [
                        '--xformers',
                        '--xformers-flash-attention'
                    ]

        # Gradio 인증 정보
        if OPTIONS['GRADIO_USERNAME'] != '':
            args += [
                f'--gradio-auth',
                OPTIONS['GRADIO_USERNAME'] +
                ('' if OPTIONS['GRADIO_PASSWORD'] ==
                    '' else ':' + OPTIONS['GRADIO_PASSWORD'])
            ]

    # 추가 인자
    args += OPTIONS['EXTRA_ARGS']

    env = {
        **os.environ,
        'HF_HOME': str(workspace / 'cache' / 'huggingface'),
    }

    # https://github.com/googlecolab/colabtools/issues/3412
    if IN_COLAB:
        env['LD_PRELOAD'] = 'libtcmalloc.so'

    try:
        execute(
            [
                OPTIONS['PYTHON_EXECUTABLE'] or 'python',
                '-u',
                '-m', 'launch',
                *args
            ],
            parser=parse_webui_output,
            cwd=str(repository),
            env=env,
            start_new_session=True,
        )
    except subprocess.CalledProcessError as e:
        if IN_COLAB and e.returncode == signal.SIGINT.value:
            raise RuntimeError(
                '프로세스가 강제 종료됐습니다, 메모리가 부족해 발생한 문제일 수도 있습니다') from e


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
