from typing import Optional, Callable, Generic, TypeVar
from ipywidgets import widgets

T = TypeVar('T')


class Option(Generic[T]):
    def __init__(
        self,
        name: Optional[str] = None,
        extractor: Optional[
            Callable[['Option'], T]
        ] = None,
    ) -> None:
        self.name = name
        self.extractor = extractor

    def selected(self):
        pass

    def deselected(self):
        pass

    def extract(self, *args, **kwargs) -> T:
        assert self.extractor
        return self.extractor(self, *args, **kwargs)


class WidgetOption(Option[T]):
    def __init__(
        self,
        widget: Optional[widgets.Widget] = None,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.widget = widget

    def selected(self):
        self.widget.layout.display = 'inherit'  # type: ignore

    def deselected(self):
        self.widget.layout.display = 'none'  # type: ignore
