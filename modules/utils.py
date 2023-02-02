import json
import time
from importlib.util import find_spec
from pathlib import Path

from . import shared
from .subprocess import call


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
