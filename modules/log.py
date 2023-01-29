from typing import Union, List, Dict, Optional, Callable, TypeVar, ClassVar
from typing_extensions import ParamSpec
from pathlib import Path
from IPython.display import display
from ipywidgets import widgets

_T = TypeVar('_T')
_A = ParamSpec('_A')


def style(style: Optional[Dict[str, str]]) -> str:
    if not style:
        return ''
    return '; '.join(f'{key}: {value}' for key, value in style.items())


class Log:
    """
    로거 또는 하위 로그 내용을 구성하는 클래스입니다
    """
    context: ClassVar[List['Log']] = []

    def __init__(
        self,
        parent: Optional['Log'] = None,
        summary: Optional[str] = None,
        style: Dict[str, str] = {},
        child_style: Dict[str, str] = {},
        only_last_lines: Optional[int] = None
    ) -> None:
        """
        로거 또는 로그를 만듭니다

        :param parent: 상위 로거
        :param widget: 로그를 렌더링할 HTML 위젯
        :param summary: 하위 로그에 대한 요약 메세지
        :param style: 로거 또는 로그 메세지의 HTML 스타일
        :param child_style: 하위 로그를 감쌀 HTML 요소의 스타일
        :param only_last_lines: 표시할 마지막 줄의 개수
        """
        self.parent = parent
        self.root_parent = None
        self.widget = None
        self.summary = summary
        self.style = None
        self.childs: List['Log'] = []
        self.child_style = {
            'padding-left': '.5em',
            **child_style
        }

        # 상위 로거가 있다면 끝에서 10줄만 보여주기
        self.only_last_lines = only_last_lines or (10 if self.parent else 0)

        # 상위 로거가 존재한다면
        if self.parent:
            # 하위 로거로 추가하기
            self.parent.childs.append(self)

            #
            self.root_parent = self.parent.root_parent or self.parent

        # 최상위 로거라면
        else:
            # 이미 사용자가 위젯을 만들었다면 표시할 필요 없음
            self.widget = widgets.HTML()

            # 위젯이 있는 최상위 로그에선 widgets.HTML 의 스타일로 사용함
            self.style = {
                **style,
                'padding': '.5em',
                'background-color': 'black',
                'line-height': '1.1',
                'color': 'white'
            }

        def wrap_context(log: Log, func: Callable[_A, _T]) -> Callable[_A, _T]:
            def wrapped(*args: _A.args, **kwargs: _A.kwargs):
                with log:
                    return func(*args, **kwargs)
            return wrapped

        self.print = wrap_context(self, Log.print)
        self.info = wrap_context(self, Log.info)
        self.warn = wrap_context(self, Log.warn)
        self.error = wrap_context(self, Log.error)

    def __enter__(self) -> 'Log':
        Log.context.insert(0, self)
        return self

    def __exit__(self, *args) -> None:
        Log.context.pop(0)

    @staticmethod
    def current_context() -> Optional['Log']:
        return Log.context[0] if len(Log.context) else None

    @staticmethod
    def print(
        message: str,
        style: Dict[str, str] = {}
    ) -> 'Log':
        log = Log.current_context()
        assert log, '컨텍스에 상위 로거가 없으면 기록할 수 없습니다'

        # if log.file:
        #     log.file.write(message)

        child_log = Log(
            parent=log,
            summary=message,
            style=style
        )

        if log.root_parent:
            log.root_parent.render()
        else:
            log.render()

        return child_log

    @staticmethod
    def info(message: str) -> 'Log':
        return Log.print(message + '\n', {'color': 'white'})

    @staticmethod
    def warn(message: str) -> 'Log':
        return Log.print(message + '\n', {'color': 'yellow'})

    @staticmethod
    def error(message: str) -> 'Log':
        return Log.print(message + '\n', {
            'font-weight': 'bold',
            'color': 'red'
        })

    def render(self) -> str:
        html = ''

        if self.summary:
            html += f'<span style="{style(self.style)}">{self.summary}</span>'

        if self.childs:
            html += f'<pre style="{style(self.child_style)}">'
            html += ''.join([
                child.render()
                for child in self.childs[-self.only_last_lines:]
            ])
            html += '</pre>'

        if self.widget:
            html = f'<div style="{style(self.style)}">{html}</div>'
            self.widget.value = html

        return html
