import itertools
import os
from pathlib import Path
from typing import Callable, Iterable, List

from ipywidgets import widgets

from modules import shared
from modules.ui import Input, Option, Placeholder, Selector
from modules.utils import alert

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
    """
    사용자가 선택한 작업 경로 하위에 있는 특정 파일을 glob 패턴을 사용해 찾은 뒤 옵션을 생성하는 함수를 반환합니다.

    :param globs: glob 패턴 목록
    :return: 옵션 생성 함수

    ```py
    from modules.ui import Selector
    from modules.workspace import workspace_lookup_generator

    Selector(workspace_lookup_generator(['*.ckpt', '*.safetensors'])
    ```
    """
    def func():
        path = Path(workspace.extract())
        path.mkdir(parents=True, exist_ok=True)

        # glob 패턴을 통해 일치하는 모든 하위 파일 목록 가져오기
        path_chunks = map(
            lambda pattern: [p for p in path.glob(pattern)],
            globs
        )

        # 2차원 배열을 1차원 배열로 펼치기
        options: List[Option] = [
            Option(str(path))
            for path in itertools.chain(*path_chunks)
        ]

        if options:
            options.insert(0, Placeholder('< 자동 선택 >'))

        return options

    return func


def mount_google_drive() -> bool:
    """
    코랩 환경에서만 구글 드라이브 마운팅을 시도합니다.

    :return: 성공 여부
    """

    if shared.IN_COLAB:
        try:
            # 마운트 후 발생하는 출력을 제거하기 위해 새 위젯 컨텍스트 만들기
            output = widgets.Output()

            with output:
                from google.colab import drive
                drive.mount(str(shared.GDRIVE_MOUNT_DIR))
                output.clear_output()

            return True

        except ImportError:
            alert('구글 드라이브에 접근할 수 없습니다, 동기화를 사용할 수 없습니다!')

    return False
