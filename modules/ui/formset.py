from typing import List, Dict

from .input import Input


class FormSet(List[Input]):
    def __init__(
        self,
        *args: Input,
        layout: Dict[str, str] = {}
    ) -> None:
        super().__init__()

        self += args
        self.layout = layout
