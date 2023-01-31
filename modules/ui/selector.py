from typing import Any, Iterable, Dict, Optional, Callable
from ipywidgets import widgets

from .utils import wrap_widget_locks
from .option import T, Option, WidgetOption


class Selector(WidgetOption[T]):
    def __init__(
        self,
        options: Iterable[Option] = [],
        refresher: Optional[
            Callable[..., Iterable[Option]]
        ] = None,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **{
            'widget': widgets.VBox(),
            **kwargs
        })

        self.options = options
        self.refresher = refresher

        self.dropdown = widgets.Dropdown(
            options=[(option.name, option) for option in options],
            # margin 이나 padding 등의 속성 때문에 전체 폭보다 조금 벗어나므로 조금 빼줌
            # border-box 를 사용하면 해결할 수 있으나 귀찮음...
            layout={'width': 'calc(100% - 5px)'}
        )

        def on_change(change: Dict[str, Any]) -> None:
            """
            옵션 값이 변경될 때 각 옵션 객체에게 이벤트를 던져주는 함수
            """
            old = change['old']
            new = change['new']

            if old and isinstance(old, Option):
                old.deselected()

            if new and isinstance(new, Option):
                new.selected()

        self.dropdown.observe(
            wrap_widget_locks(on_change, [self.dropdown]),
            names='value'  # type: ignore
        )

        self.refresh()

    def refresh(self) -> None:
        children = []

        if self.refresher:
            self.dropdown.options = tuple([
                (opt.name, opt)
                for opt in list(self.refresher()) + list(self.options)
            ])

            button = widgets.Button(
                description='🔄',
                layout={'width': 'auto'})

            button.on_click(lambda _: self.refresh())

            children.append(
                widgets.GridBox(
                    (widgets.Box((self.dropdown,)), button),
                    layout={'grid_template_columns': '5fr 1fr'}
                )
            )

        else:
            children.append(self.dropdown)

        assert isinstance(self.dropdown.options, tuple)
        for _, opt in self.dropdown.options:
            # 위젯이 있는 옵션이라면 위젯 집합에 추가하기
            if isinstance(opt, WidgetOption):
                children.append(opt.widget)

            event = opt.selected if self.dropdown.value == opt else opt.deselected
            event()

        # 각 옵션 값 (비)활성화 이벤트 실행
        assert isinstance(self.widget, widgets.VBox)
        self.widget.children = tuple(children)

    def extract(self, *args, **kwargs) -> Optional[T]:
        assert isinstance(self.dropdown.value, Option), ''
        return self.dropdown.value.extract(*args, **kwargs)
