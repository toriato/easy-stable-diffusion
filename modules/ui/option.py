from typing import Callable, Generic, Optional, TypeVar

from ipywidgets import widgets

T = TypeVar('T')


class Option(Generic[T]):
    """
    사용자에 의해 선택 됐을 때 미리 설정한 값을 반환하는 옵션입니다.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        extractor: Optional[
            Callable[['Option'], T]
        ] = None,
    ) -> None:
        """
        :param name: 사용자에게 보여줄 이름, None 일 경우 'None' 으로 보입니다.
        :param extractor: 옵션에서 값을 추출하는 함수, None 일 경우 런타임 assert 를 통과하지 못하고 오류를 반환합니다.
        """
        self.name = name
        self.extractor = extractor

    def selected(self):
        """
        Selector 등의 하위 요소에서 선택됐을 때 실행될 이벤트 함수입니다.
        """
        pass

    def deselected(self):
        """
        Selector 등의 하위 요소에서 선택 해제됐을 때 실행될 이벤트 함수입니다.
        """
        pass

    def extract(self, *args, **kwargs) -> T:
        """
        사용자가 미리 지정한 함수로부터 값을 꺼내와 반환합니다.

        :return: 사용자가 미리 지정한 함수로부터 반환된 값
        """
        assert self.extractor, '값을 추출할 함수가 선언되지 않았습니다'
        return self.extractor(self, *args, **kwargs)


class WidgetOption(Option[T]):
    """
    IPython 의 위젯을 구현하는 옵션입니다.
    `extractor` 인자가 없다면 위젯의 `value` 속성을 반환합니다.
    """

    def __init__(
        self,
        widget: Optional[widgets.Widget] = None,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        assert widget, '위젯 요소 없이 위젯 옵션을 만들 수 없습니다'
        self.widget = widget

    def selected(self):
        self.widget.layout.display = 'inherit'  # type: ignore

    def deselected(self):
        self.widget.layout.display = 'none'  # type: ignore

    def extract(self, *args, **kwargs) -> T:
        return self.widget.value  # type: ignore
