from pathlib import Path
from typing import Callable

from modules.script_callbacks import on_app_started

from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.responses import FileResponse
from gradio.routes import App as GradioApp

app: GradioApp

# 이 값이 담고 있는 함수는 다음과 같음:
# https://github.com/gradio-app/gradio/blob/58b1a074ba342fe01445290d680a70c9304a9de1/gradio/routes.py#L249-L274
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
    p = Path(path)
    if Path(app.cwd).absolute() in p.absolute().parents:
        return FileResponse(
            p.resolve(), headers={"Accept-Ranges": "bytes"}
        )

    # 접근할 수 있는 경로가 아니라면 오류 그대로 반환하기
    raise original_error


def hook(_, app_instance: FastAPI):
    global app, original_endpoint

    if not isinstance(app_instance, GradioApp):
        raise ValueError()

    app = app_instance

    for route in app.router.routes:
        if not isinstance(route, APIRoute):
            continue

        # Gradio 에서 내부 파일에 접근하는 경로의 패턴
        # https://github.com/gradio-app/gradio/blob/58b1a074ba342fe01445290d680a70c9304a9de1/gradio/routes.py#L248
        if route.path != '/file={path:path}':
            continue

        if route.dependant.call != endpoint:
            original_endpoint = route.dependant.call
            route.dependant.call = endpoint

        break


on_app_started(hook)
