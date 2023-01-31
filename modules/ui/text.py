from .option import Option


class Text(Option[str]):
    def __init__(self, name: str, *args, **kwargs) -> None:
        super().__init__(name, *args, **kwargs)

    def extract(self) -> str:
        assert self.name
        return self.name
