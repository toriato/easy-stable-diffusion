import itertools

from typing import Iterable, List, Callable
from pathlib import Path

from modules.ui import Selector, SelectorOption

PYTHON_EXECUTABLE = 'python3.10'

REPO_DIR = Path(__file__).parent.joinpath('repository')

workspace = Selector(
    options=[
        SelectorOption('SD')
    ]
)


def workspace_lookup_generator(
    globs: List[str]
) -> Callable[..., Iterable[SelectorOption]]:
    # TODO: 기본 작업 경로는 레포지토리를 가르켜야함
    workspace_path = workspace.extract() or REPO_DIR

    def func():
        # glob 패턴을 통해 일치하는 모든 하위 파일 목록 가져오기
        path_chunks = map(
            lambda pattern: [p for p in workspace_path.glob(pattern)],
            globs
        )

        # 2차원 배열을 1차원 배열로 펼치기
        options = [
            SelectorOption(str(path))
            for path in itertools.chain(*path_chunks)
        ]

        if options:
            options.insert(0, SelectorOption('< 자동 선택 >', lambda _: None))

        return options

    return func
