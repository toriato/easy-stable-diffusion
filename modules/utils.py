import json
import shlex
import subprocess
import time
from importlib.util import find_spec
from pathlib import Path
from shutil import copyfileobj, which
from typing import Generic, Optional, TypeVar

import requests

from . import shared
from .log import Log
from .subprocess import call

T = TypeVar('T')


class NullContextManager(Generic[T]):
    def __init__(self, dummy: T = None):
        self.dummy = dummy

    def __enter__(self) -> T:
        return self.dummy

    def __exit__(self, *args):
        pass


def alert(text: str, unassign=False) -> None:
    """
    자바스크립트의 alert() 메소드를 사용해 사용자 브라우저에 메세지를 표시합니다

    :param text: 표시할 내용
    :param unassign: 코랩 환경에서 런타임을 해제할지?
    """
    try:
        from IPython.display import display
        from ipywidgets import widgets
        display(widgets.HTML(f'<script>alert({json.dumps(text)})</script>'))
    except ImportError:
        pass

    if unassign:
        try:
            from google.colab import runtime
            time.sleep(2)
            runtime.unassign()
        except ImportError:
            pass


def has_python_package(pkg_name: str, executable: Optional[str] = None) -> bool:
    """
    현재 또는 특정 Python 환경의 패키지 존재 여부를 반환합니다.

    :param pkg_name: 패키지 이름
    :param executable: 패키지를 검사할 Python 실행 파일 경로

    :return: 패키지 존재 여부
    """
    if not executable:
        return find_spec(pkg_name) is not None

    try:
        call([
            executable, '-c',
            f'''
            import importlib
            import sys
            sys.exit(0 if importlib.util.find_spec({shlex.quote(pkg_name)}) else 1)
            '''
        ])
    except subprocess.CalledProcessError:
        return False

    return True


def mount_google_drive() -> Path:
    """
    코랩 환경에서만 구글 드라이브 마운팅을 시도합니다.

    :return: 마운팅된 경로
    """
    from google.colab import drive

    # 마운트 후 발생하는 출력을 제거하기 위해 새 위젯 컨텍스트 만들기
    output = None

    try:
        from ipywidgets import widgets
        output = widgets.Output()
    except ImportError:
        pass

    if output:
        with output:
            drive.mount(str(shared.GDRIVE_MOUNT_DIR))
            output.clear_output()
    else:
        drive.mount(str(shared.GDRIVE_MOUNT_DIR))

    return shared.GDRIVE_MOUNT_DIR


def hook_runtime_disconnect():
    """
    셀 실행 후 런타임을 자동으로 종료하도록 후킹합니다
    """

    # google.colab 패키지가 없으면 ImportError 를 raise 하므로
    # 코랩 런타임 환경 밖에서 이 코드는 동작하지 않음
    from google.colab import runtime

    # asyncio 는 여러 겹으로 사용할 수 없게끔 설계됐기 때문에
    # 주피터 노트북 등 이미 루프가 돌고 있는 곳에선 사용할 수 없음
    # 이는 nest-asyncio 패키지를 통해 어느정도 우회하여 사용할 수 있음
    # https://pypi.org/project/nest-asyncio/
    if not find_spec('nest_asyncio'):
        call(['pip', 'install', 'nest-asyncio'])

    import nest_asyncio
    nest_asyncio.apply()

    async def unassign():
        runtime.unassign()

    # 평범한 환경에선 비동기로 동작하여 바로 실행되나
    # 코랩? IPython 환경에선 순차적으로 실행되기 때문에 현재 셀 종료 후 즉시 실행됨
    import asyncio
    asyncio.create_task(unassign())


def download(
    url: str,
    target: str,
    summary: Optional[str] = None,
    ignore_aria2=False,
    *args, **kwargs
) -> None:
    """
    파일을 다운로드합니다

    :param url: 다운로드할 파일의 URL
    :param target: 다운로드할 파일의 경로
    :param ignore_aria2: aria2 를 사용하지 않을지?
    :param args: 다운로드 메소드에 전달할 인자
    :param kwargs: 다운로드 메소드에 전달할 키워드 인자
    """

    with Log.info('파일을 다운로드합니다.') if summary is None else NullContextManager():

        # 파일을 받을 디렉터리 만들기
        Path(target).parent.mkdir(0o777, True, True)

        # 빠른 다운로드를 위해 aria2 패키지 설치 시도하기
        if not ignore_aria2:
            if which('apt') is not None and not which('aria2c'):
                call(['apt', 'install', 'aria2'], *args, **kwargs)

            call(
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
                *args, **kwargs)

        elif which('curl'):
            call(
                [
                    'curl',
                    '--location',
                    '--output', target,
                    url
                ],
                *args, **kwargs)

        else:
            if 'summary' in kwargs.keys():
                Log.info(kwargs.pop('summary'), **kwargs)

            with requests.get(url, stream=True, *args, **kwargs) as res:
                res.raise_for_status()

                with open(target, 'wb') as file:
                    # 받아온 파일 디코딩하기
                    # https://github.com/psf/requests/issues/2155#issuecomment-50771010
                    import functools
                    res.raw.read = functools.partial(
                        res.raw.read,
                        decode_content=True)

                    # TODO: 파일 길이가 적합한지?
                    copyfileobj(res.raw, file, length=16*1024*1024)
