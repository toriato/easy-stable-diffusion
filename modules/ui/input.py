from typing import TypeVar

from ipywidgets import widgets

from .option import WidgetOption

T = TypeVar('T', bound=str)


class Input(WidgetOption[T]):
    """
    사용자로부터 텍스트를 받아오는 옵션입니다.
    """

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

    def extract(self, *args, **kwargs) -> T:
        if self.extractor:
            return self.extractor(self, *args, **kwargs)

        assert isinstance(self.widget, widgets.Text)
        assert isinstance(
            self.widget.value,
            str), '사용자 추출 함수가 없다면 위젯은 항상 문자열을 반환해야 합니다'

        # TypeVar 에 기본 자료형을 설정할 수 있으면 얼마나 좋을까...?
        # https://peps.python.org/pep-0696/
        return self.widget.value  # type: ignore
