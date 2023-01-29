from typing import Iterable, Callable


def wrap_widget_locks(
    callback: Callable,
    widgets: Iterable = ()
) -> Callable:
    """
    콜백 전 모든 위젯을 비활성화하고 끝나면 마지막 상태로 되돌립니다

    :param callback: 실행할 함수
    :param widgets: 비활성화할 위젯들

    :return: 감싸진 함수
    """
    def wrapped(*args, **kwargs):
        original_states = {}
        for widget in widgets:
            if hasattr(widget, 'disabled'):
                original_states[widget] = widget.disabled
                widget.disabled = True

        try:
            callback(*args, **kwargs)
        finally:
            for widget, state in original_states.items():
                widget.disabled = state

    return wrapped
