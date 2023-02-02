import json
import time
from pathlib import Path

from . import shared


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
