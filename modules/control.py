from typing import Optional, Iterable, List, Dict, Callable, Generic, TypeVar
from ipywidgets import widgets

from modules.ui import Option, WidgetOption

T = TypeVar('T')

ControlContext = Dict[str, 'Control']


class Control(Generic[T]):
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

    def extract(self, context: ControlContext) -> Optional[T]:
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
