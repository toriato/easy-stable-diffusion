from typing import Union, List, Dict, Optional, Callable, TypeVar, ClassVar
from typing_extensions import ParamSpec
from pathlib import Path

_T = TypeVar('_T')
_A = ParamSpec('_A')


class Log:
    """
    로거 또는 하위 로그 내용을 구성하는 클래스입니다
    """
    context: ClassVar[Optional['Log']] = None

    def __init__(
        self,
        path_or_text: Union[Path, str],
        parent: Optional['Log'] = None,
        widget: Optional[object] = None,
        style: Dict[str, str] = {},
        child_style: Dict[str, str] = {},
        only_last_lines: Optional[int] = None
    ) -> None:
        """
        로거 또는 로그를 만듭니다

        :param path_or_text: 로그 파일의 경로 또는 로그 내용
        :param parent: 상위 로거
        :param widget: 로그를 렌더링할 HTML 위젯
        :param style: 로거 또는 로그 메세지의 HTML 스타일
        :param child_style: 하위 로그를 감쌀 HTML 요소의 스타일
        :param only_last_lines: 표시할 마지막 줄의 개수
        """
        self.parent = parent or Log.context
        if self.parent:
            self.parent.childs.append(self)

        self.path = None
        self.file = None
        self.widget = None

        self.text = None
        self.style = None

        # 경로를 받으면 최상위 로그로 간주하기
        if isinstance(path_or_text, Path):
            self.path = path_or_text
            self.file = path_or_text.open('w')

            try:
                from IPython.display import display
                from ipywidgets import widgets

                # 이미 사용자가 위젯을 만들었다면 표시할 필요 없음
                if widget:
                    assert isinstance(widget, widgets.HTML)
                    self.widget = widget

                # 기본 위젯 생성한 뒤 표시하기
                else:
                    self.widget = widgets.HTML()
                    display(self.widget)

            except ImportError:
                pass

        # 최상위 로거가 아닌 하위 로거거나 일반 로그인 경우
        else:
            assert not widget, '위젯은 최상위 로거에서만 사용할 수 있습니다'
            assert self.parent, '상위 로거 없이 로그 메세지를 기록할 수 없습니다'
            self.text = path_or_text

        # 위젯이 있는 최상위 로그에선 widgets.HTML 의 스타일로 사용함
        if self.widget:
            self.style = {
                **style,
                'padding': '.5em',
                'background-color': 'black',
                'line-height': '1.1',
                'color': 'white'
            }

        self.childs: List[Log] = []
        self.child_style = {
            'padding-left': '.5em',
            **child_style
        }

        # 상위 로거가 있다면 끝에서 10줄만 보여주기
        if only_last_lines is None:
            only_last_lines = 10 if self.parent else 0

        self.only_last_lines = only_last_lines

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
        Log.context = self
        return self

    def __exit__(self, *args) -> None:
        Log.context = None

    @staticmethod
    def print(
        message: str,
        style: Dict[str, str] = {}
    ) -> 'Log':
        log = Log.context
        assert log, '컨텍스에 상위 로거가 없으면 기록할 수 없습니다'

        if not log.widget:
            print(message, end='')

        if log.file:
            log.file.write(message)

        child_log = Log(
            message,
            parent=log,
            style=style
        )
        log.render(True)

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

    def render(self, recursive=False) -> Optional[str]:
        def style(style: Optional[Dict[str, str]]) -> str:
            if not style:
                return ''
            return '; '.join(f'{key}: {value}' for key, value in style.items())

        if self.parent and recursive:
            self.parent.render(recursive)

        html = ''

        if self.text:
            html += f'<span style="{style(self.style)}">{self.text}</span>'

        if self.childs:
            html += f'<pre style="{style(self.child_style)}">'
            html += ''.join([
                html
                for html in [
                    child.render()
                    for child in self.childs[-self.only_last_lines:]
                ]
                if html
            ])
            html += '</pre>'

        if self.widget:
            html = f'<div style="{style(self.style)}">{html}</div>'
            self.widget.value = html

        return html
