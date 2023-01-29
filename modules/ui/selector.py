from typing import Any, Iterable, List, Dict, Optional, Callable
from pathlib import Path
from ipywidgets import widgets

from .utils import wrap_widget_locks


class SelectorOption:
    def __init__(
        self,
        name: str,
        extractor: Optional[
            Callable[['SelectorOption'], Any]
        ] = None,
    ) -> None:
        self.name = name
        self.extractor = extractor

    def create(self, selector: 'Selector') -> Optional[widgets.Widget]:
        pass

    def selected(self, selector: 'Selector'):
        pass

    def deselected(self, selector: 'Selector'):
        pass

    def extract(self) -> Any:
        if self.extractor:
            return self.extractor(self)
        return self.name


class SelectorWidget(SelectorOption):
    def __init__(
        self,
        name: str,
        widget: widgets.Widget,
        extractor: Optional[Callable[['SelectorOption'], Any]] = None
    ) -> None:
        super().__init__(name, extractor)
        self.widget = widget

    def selected(self, _):
        self.widget.layout.display = 'inherit'  # type: ignore

    def deselected(self, _):
        self.widget.layout.display = 'none'  # type: ignore


class SelectorText(SelectorWidget):
    def __init__(
        self,
        name='< 직접 입력 >',
        default_text='',
        extractor: Optional[Callable[['SelectorText'], str]] = None,
    ) -> None:
        super().__init__(
            name,
            widgets.Text(
                value=default_text,
                layout={'width': 'auto'}
            )
        )

        self.extractor = extractor

    def extract(self) -> str:
        if self.extractor:
            return self.extractor(self)

        assert isinstance(self.widget, widgets.Text)
        assert isinstance(self.widget.value, str)
        return self.widget.value


class Selector:
    """
    로컬 파일 시스템 또는 인터넷으로부터 파일을 선택하거나 업로드할 수 있는 위젯 집합을 만듭니다
    """

    def __init__(
        self,
        options: Iterable[SelectorOption] = [],
        refresher: Optional[
            Callable[..., Iterable[SelectorOption]]
        ] = None,
    ) -> None:
        """
        :param options: 드롭다운 추가할 옵션들
        :param refresher: 파일 검색에 사용할 함수
        """
        self.dropdown = widgets.Dropdown(
            options=[(option.name, option) for option in options],
            # margin 이나 padding 등의 속성 때문에 전체 폭보다 조금 벗어나므로 조금 빼줌
            # border-box 를 사용하면 해결할 수 있으나 귀찮음...
            layout={'width': 'calc(100% - 5px)'}
        )

        self.refresh_button = widgets.Button(
            description='🔄',
            disabled=True,
            layout={'width': 'auto'})

        self.lock_group = [self.dropdown, self.refresh_button]

        def on_update_dropdown(change: Dict[str, Any]) -> None:
            """
            옵션 값이 변경될 때 각 옵션 객체에게 이벤트를 던져주는 함수
            """
            old = change['old']
            new = change['new']

            if old and isinstance(old, SelectorOption):
                old.deselected(self)

            if new and isinstance(new, SelectorOption):
                new.selected(self)

        def on_click_refresh_button(_) -> None:
            if refresher:
                self.dropdown.options = tuple([
                    (opt.name, opt)
                    for opt in list(refresher()) + list(options)
                ])

            # 각 옵션 값 (비)활성화 이벤트 실행
            assert isinstance(self.dropdown.options, tuple)
            for _, opt in self.dropdown.options:
                if self.dropdown.value == opt:
                    opt.selected(self)
                else:
                    opt.deselected(self)

        # 각 위젯에 이벤트 핸들러 연결
        self.dropdown.observe(
            wrap_widget_locks(
                on_update_dropdown,
                self.lock_group
            ),
            names='value'  # type: ignore
        )

        if refresher:
            self.refresh_button.disabled = False
            self.refresh_button.on_click(
                wrap_widget_locks(
                    on_click_refresh_button,
                    self.lock_group
                ))

        on_click_refresh_button(None)

    def create_ui(self) -> widgets.Box:
        """
        위젯 집합을 담고 있는 박스 위젯을 만듭니다
        """
        option_widgets = [
            opt.widget
            for _, opt in self.dropdown.options  # type: ignore
            if isinstance(opt, SelectorWidget)
        ]

        return widgets.VBox((
            widgets.GridBox(
                (widgets.Box((self.dropdown,)), self.refresh_button),
                layout={'grid_template_columns': '5fr 1fr'}
            ),
            *option_widgets
        ))

    def extract(self) -> Optional[Path]:
        """
        사용자가 선택한 파일의 경로를 가져옵니다

        Returns: 사용자가 선택한 파일의 문자열 경로
        """
        assert isinstance(self.dropdown.value, SelectorOption), ''

        path = self.dropdown.value.extract()
        if path is None:
            return None

        return Path(path)
