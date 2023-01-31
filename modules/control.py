from typing import Callable, Dict, Generic, Iterable, List, Optional, TypeVar

from ipywidgets import widgets

from modules.ui import Option, WidgetOption

T = TypeVar('T')

ControlContext = Dict[str, 'Control']


class Control(Generic[T]):
    """
    Option 요소를 감싸 프로그램에 전달한 인자나 컨텍스트에서 사용될 고유한 키를 구성합니다.
    또한 유저 인터페이스에서 HTML 설명문을 추가할 수 있습니다.
    """

    def __init__(
        self,
        option: Option[T],
        key: Optional[str] = None,
        argument: Optional[str] = None,
        summary_html: Optional[str] = None,
        layout: Dict[str, str] = {},
        extractor: Optional[
            Callable[['Control', 'ControlContext'], T]
        ] = None
    ) -> None:
        """
        :param option: 감쌀 Option 요소
        :param key: 컨텍스트에서 사용될 고유한 키, None 이면 컨텍스트로써 사용하지 않음
        :param argument: 프로그램에 전달될 인자 이름, None 이면 인자로써 사용하지 않음
        :param summary_html: 유저 인터페이스에 추가될 HTML 설명문
        :param layout: 전체 요소의 레이아웃
        :param extractor: 컨텍스트에서 값을 추출하는 함수, None 이면 Option 의 추출 함수를 사용합니다
        """
        self.option = option
        self.key = key
        self.argument = argument
        self.summary_html = summary_html
        self.extractor = extractor
        self.wrapper = None

        if isinstance(option, WidgetOption):
            children = [option.widget]

            if summary_html:
                children.insert(0, widgets.HTML(summary_html))

            self.wrapper = widgets.VBox(
                children,
                layout={
                    'padding': '.5em',
                    'border': '2px solid black',
                    **layout
                }
            )

    def extract(self, context: ControlContext) -> T:
        if self.extractor:
            return self.extractor(self, context)
        return self.option.extract()


class ControlGrid(List[Control]):
    def __init__(
        self,
        controls: Iterable[Control],
        layout: Optional[Dict[str, str]] = None
    ) -> None:
        super().__init__(controls)
        self.layout = layout or {}
