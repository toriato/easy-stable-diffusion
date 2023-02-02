from typing import Any, Callable, Dict, Iterable, Optional

from ipywidgets import widgets

from .option import Option, T, WidgetOption
from .utils import wrap_widget_locks


class Selector(WidgetOption[T]):
    """
    사용자가 선택한 하위 옵션들로부터 값을 가져오는 옵션입니다.
    """

    def __init__(
        self,
        options: Iterable[Option] = [],
        refresher: Optional[
            Callable[..., Iterable[Option]]
        ] = None,
        *args, **kwargs
    ) -> None:
        """
        :param options: 새로고침해도 사라지지 않을 기본 옵션들
        :param refresher: 옵션을 새로고침할 때 추가될 옵션을 반환하는 함수, None 이라면 새로고침 버튼이 나타나지 않습니다.
        """
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
            선택한 옵션이 변경될 때 모든 옵션에게 각 상태에 맞는 이벤트를 던져주는 이벤트 함수입니다.
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
        """
        사용자가 선택할 수 있는 옵션을 새로고치는 함수입니다. `refresher` 함수 존재 여부에 따라 새로고침 버튼을 보여주거나 숨기기도 합니다.
        """
        children = []

        if self.refresher:
            self.dropdown.options = tuple([
                (opt.name, opt)

                # 새로 추가된 옵션은 항상 기본 옵션보다 앞으로 가게끔 정렬
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

            # 각 위젯의 상태에 맞는 이벤트 함수 실행하기
            event = opt.selected if self.dropdown.value == opt else opt.deselected
            event()

        # 각 옵션 값 (비)활성화 이벤트 실행
        assert isinstance(self.widget, widgets.VBox)
        self.widget.children = tuple(children)

    def extract(self, *args, **kwargs) -> T:
        """
        사용자가 선택한 옵션으로부터 값을 가져오는 함수입니다.
        """
        assert isinstance(self.dropdown.value, Option), ''
        return self.dropdown.value.extract(*args, **kwargs)
