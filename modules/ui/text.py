from typing import TypeVar
from ipywidgets import widgets

from .option import WidgetOption

T = TypeVar('T', bound=str)


class Text(WidgetOption[T]):
    def __init__(
        self,
        default_text='',
        *args, **kwargs
    ) -> None:
        super().__init__(
            *args, **{
                'widget': widgets.Text(
                    value=default_text,
                    layout={'width': 'auto'}
                ),
                ** kwargs
            }
        )

    def extract(self, *args, **kwargs):
        if self.extractor:
            return self.extractor(self, *args, **kwargs)

        assert isinstance(self.widget, widgets.Text)
        assert isinstance(
            self.widget.value,
            str), '사용자 추출 함수가 없다면 위젯은 항상 문자열을 반환해야 합니다'

        return self.widget.value
