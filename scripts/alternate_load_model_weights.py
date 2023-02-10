import os
import sys
import threading
import time

from modules import call_queue, paths, scripts, sd_models, shared

sys.path.insert(0, paths.data_path)


def on_app_started(*args, **kwargs):
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


scripts.script_callbacks.on_app_started(on_app_started)
