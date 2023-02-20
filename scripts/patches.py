import os
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.routing import APIRoute
from gradio import Blocks
from modules import paths, shared
from modules.script_callbacks import on_app_started


class Patches:
    def __init__(self, demo: Optional[Blocks], app: FastAPI) -> None:
        self.demo = demo
        self.app = app

        self.patch_gradio_route()
        self.patch_data_dir_path()

    def patch_gradio_route(self):
        """
        Gradio 에서 `/file={path} 경로가 앱 경로와 다른 장치에 위치할 때
        `Path.resolve().parents` 값 사용으로 인해 하위 디렉터리가 아닌 것으로 인식해
        접근할 수 없는 이슈를 해결하고자 기존 엔드포인트 함수를 재정의합니다. 
        """
        original_endpoint: Optional[Callable] = None

        async def endpoint(path: str, *args, **kwargs):
            original_error: ValueError

            try:
                assert original_endpoint
                return await original_endpoint(path, *args, **kwargs)
            except ValueError as e:
                # `path` 가 `app.cwd` 속에 있는 경로가 아닌 경우에 ValueError 를 반환함
                # https://github.com/gradio-app/gradio/blob/58b1a074ba342fe01445290d680a70c9304a9de1/gradio/routes.py#L272
                original_error = e

            # `Path.resolve()` 사용으로 인해 `app.cwd` 내에 있는 심볼릭 링크의 경우 ValueError 를 반환할 수 있음
            # https://github.com/gradio-app/gradio/blob/58b1a074ba342fe01445290d680a70c9304a9de1/gradio/routes.py#L263-L270
            parents = Path(path).absolute().parents

            if Path(shared.data_path).absolute() not in parents:
                raise original_error

            return FileResponse(path, headers={'Accept-Ranges': 'bytes'})

        for route in self.app.router.routes:
            if not isinstance(route, APIRoute):
                continue

            # Gradio 에서 내부 파일에 접근하는 경로의 패턴
            # https://github.com/gradio-app/gradio/blob/58b1a074ba342fe01445290d680a70c9304a9de1/gradio/routes.py#L248
            if route.path != '/file={path:path}':
                continue

            if route.dependant.call != original_endpoint:
                original_endpoint = route.dependant.call  # type: ignore
                route.dependant.call = endpoint

            break

    def patch_data_dir_path(self):
        """
        `--data-dir` 인자를 사용하면 `extensions/` 등의 위치를 Import 하는 확장 기능들이 망가져버리므로
        시스템 PATH 에 `--data-dir` 경로를 추가합니다.
        """
        sys.path.insert(0, paths.data_path)


on_app_started(
    lambda demo, app: Patches(demo, app)
)
