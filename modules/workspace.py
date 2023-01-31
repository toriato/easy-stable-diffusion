import os
import itertools

from typing import Iterable, List, Callable
from pathlib import Path
from ipywidgets import widgets

from modules import shared
from modules.alert import alert
from modules.ui import Selector, Option, Text, Placeholder, Input


prepend_options = []

if shared.IN_COLAB:
    prepend_options.append(
        Input(
            name='< 구글 드라이브 >',
            default_text='SD',
            extractor=lambda option: os.path.join(
                shared.GDRIVE_DIR,
                str(option.widget.value)  # type: ignore
            )
        )
    )

workspace = Selector[str](
    options=[
        *prepend_options,
        Input(
            name='< 로컬 파일 >',
            default_text='SD',
            extractor=lambda option: os.path.join(
                shared.APP_DIR,
                str(option.widget.value)  # type: ignore
            )
        )
    ]
)


def workspace_lookup_generator(
    globs: List[str]
) -> Callable[..., Iterable[Option]]:
    # TODO: 기본 작업 경로는 레포지토리를 가르켜야함

    path = Path(workspace.extract())

    def func():
        # glob 패턴을 통해 일치하는 모든 하위 파일 목록 가져오기
        path_chunks = map(
            lambda pattern: [p for p in path.glob(pattern)],
            globs
        )

        # 2차원 배열을 1차원 배열로 펼치기
        options: List[Option] = [
            Text(str(path))
            for path in itertools.chain(*path_chunks)
        ]

        if options:
            options.insert(0, Placeholder('< 자동 선택 >'))

        return options

    return func


def mount_google_drive() -> bool:
    """
    코랩 환경에서 구글 드라이브 마운팅을 시도합니다
    """

    if shared.IN_COLAB:
        try:
            # 마운트 후 발생하는 출력을 제거하기 위해 새 위젯 컨텍스 만들기
            output = widgets.Output()

            with output:
                from google.colab import drive
                drive.mount(str(shared.GDRIVE_MOUNT_DIR))
                output.clear_output()

            return True

        except ImportError:
            alert('구글 드라이브에 접근할 수 없습니다, 동기화를 사용할 수 없습니다!')

    return False
