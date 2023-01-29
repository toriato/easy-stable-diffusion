import os
import itertools

from typing import Iterable, List, Callable

from modules import shared
from modules.ui import Selector, SelectorOption, SelectorText


prepend_options = [
    SelectorText(
        '< 로컬 파일 시스템 >',
        'SD',
        lambda option: os.path.join(
            shared.APP_DIR,
            str(option.widget.value)
        )
    )
]

if shared.IN_COLAB:
    prepend_options.append(
        SelectorText(
            '< 구글 드라이브 동기화 >',
            'SD',
            lambda option: os.path.join(
                shared.GDRIVE_DIR,
                str(option.widget.value)
            )
        )
    )

workspace = Selector(prepend_options)


def workspace_lookup_generator(
    globs: List[str]
) -> Callable[..., Iterable[SelectorOption]]:
    # TODO: 기본 작업 경로는 레포지토리를 가르켜야함
    workspace_path = workspace.extract() or shared.REPO_DIR

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
