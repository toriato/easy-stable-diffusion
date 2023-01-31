from .option import Option


class Placeholder(Option[str]):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def extract(self) -> None:
        return None
