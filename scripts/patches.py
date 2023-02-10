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
from modules import call_queue, paths, sd_models, shared
from modules.script_callbacks import on_app_started


class Patches:
    def __init__(self, demo: Optional[Blocks], app: FastAPI) -> None:
        self.demo = demo
        self.app = app

        self.patch_gradio_route()
        self.patch_model_change_event()
        self.patch_data_dir_path()

    def patch_gradio_route(self):
        """
        Gradio 에서 `/file={path} 경로가 앱 경로와 다른 장치에 위치할 때
        `Path.resolve().parents` 값 사용으로 인해 하위 디렉터리가 아닌 것으로 인식해
        접근할 수 없는 이슈를 해결하고자 기존 엔드포인트 함수를 재정의합니다. 
        """
        original_endpoint: Callable

        async def endpoint(path: str, *args, **kwargs):
            original_error: ValueError

            try:
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
                endpoint = route.dependant.call  # type: ignore
                route.dependant.call = endpoint

            break

    def patch_data_dir_path(self):
        """
        `--data-dir` 인자를 사용하면 `extensions/` 등의 위치를 Import 하는 확장 기능들이 망가져버리므로
        시스템 PATH 에 `--data-dir` 경로를 추가합니다.
        """
        sys.path.insert(0, paths.data_path)

    def patch_model_change_event(self):
        """
        현재 가용할 수 있는 메모리보다 큰 모델 파일을 불러오려고 할 때
        프로세스를 재시작 할 수 있도록 현재 프로세스를 정상 죵료하도록 모델 선택 컴포넌트의 이벤트를 수정합니다.
        """

        # TODO: VAE 같은 다른 모델 선택할 때도 메모리 밀어줘야함
        def on_change():
            # 먼저 설정 파일을 저장해둬야 메모리 부족으로 터져도 다시 불러올 수 있음
            shared.opts.save(shared.config_filename)

            meminfo = dict(
                (i.split()[0].rstrip(':'), int(i.split()[1]))
                for i in open('/proc/meminfo').readlines()
            )

            # 사용 가능한 메모리가 충분할 때만 모델 불러오기
            if 4 < meminfo['MemAvailable'] / 1024 / 1024:
                return call_queue.wrap_queued_call(
                    lambda: sd_models.reload_model_weights()
                )()

            # 클라이언트에게 결과를 반환하지 않으면 설정을 다시 바꿀 수 없게 되어버림
            # 새 스레드에서 1초 대기 후 프로세스를 종료해 인터페이스가 먹통되지 않도록 우회함
            def _exit():
                time.sleep(1)
                os._exit(0)

            threading.Thread(target=_exit).start()

        shared.opts.onchange('sd_model_checkpoint', on_change, call=False)


on_app_started(
    lambda demo, app: Patches(demo, app)
)
