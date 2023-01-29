from typing import Any, Dict, Optional, Callable
from ipywidgets import widgets

InputContext = Dict[str, 'Input']


class Input:
    def __init__(
        self,
        name: Optional[str] = None,
        widget: Optional[widgets.Widget] = None,
        summary_html: Optional[str] = None,
        extractor: Optional[Callable[['Input', InputContext], Any]] = None,
        validator: Optional[Callable[['Input', InputContext], bool]] = None
    ) -> None:
        """
        스크립트의 인자를 구성하는 클래스입니다

        :param name: 스크립트에서 사용하는 인자 이름, (예: --camel_case_name)
        :param widget: 인터페이스에 표시되는 위젯 개체
        :param summary_html: 위젯 상단에 표시되는 설명 HTML
        :param extractor: 위젯으로부터 값을 추출하는 함수
        :param validator: 위젯의 값이 유효한지 검사하는 함수
        """

        if widget and not widget.layout.width:  # type: ignore
            widget.layout.width = 'calc(100% - 5px)'  # type: ignore

        self.widget = widget
        self.name = name
        self.summary_html = summary_html
        self.extractor = extractor
        self.validator = validator

    def create_ui(self) -> widgets.VBox:
        return widgets.VBox((
            widgets.HTML(self.summary_html),
            self.widget,
        ))

    def extract(self, context: InputContext) -> Any:
        if not self.extractor:
            return self.widget.value if self.widget else None  # type: ignore

        return self.extractor(self, context)
