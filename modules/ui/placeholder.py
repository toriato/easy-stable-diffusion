from .option import Option


class Placeholder(Option[str]):
    """
    아무런 작업도 하지 않고 단순히 None 을 반환하는 옵션입니다.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def extract(self) -> None:
        return None
