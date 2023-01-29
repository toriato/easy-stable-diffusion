import time
import json

from modules.log import Log


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
