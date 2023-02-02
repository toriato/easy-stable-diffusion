import shlex
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

from modules.log import Log


def call(
    *args,
    fold_on_success=True,
    unfold_on_failed=True,
    **kwargs
) -> int:
    """
    하위 프로세스를 실행하고 모든 출력을 print() 함수로 전달합니다

    :param fold_on_success: 성공시 출력을 접습니다
    :param unfold_on_failed: 실패시 출력을 펼칩니다
    :param args: subprocess.Popen() 함수에 전달할 인자들
    :param kwargs: subprocess.Popen() 함수에 전달할 인자들

    :return: 프로세스의 종료 코드
    """
    p = subprocess.Popen(*args, **{
        'stdout': subprocess.PIPE,
        'stderr': subprocess.STDOUT,
        'encoding': 'utf-8',
        **kwargs
    })

    # 사용자에게 받은 명령어 및 인자 조합을 문자열로 변환합니다
    # TODO: 모든 subprocess._CMD 자료형을 지원하는지 검증 필요
    if isinstance(p.args, str):
        raw_cmd = p.args
    else:
        raw_cmd = shlex.join(p.args)  # type: ignore

    log = Log(
        Log.current_context(),
        raw_cmd,
        style={'color': 'yellow'},
        child_style={'color': 'gray'}
    )

    while p.poll() is None:
        assert p.stdout
        line = p.stdout.readline()
        if not line:
            continue

        log.print(line)

    rc = p.poll()
    assert rc is not None

    # 로그 스타일 업데이트
    try:
        if rc != 0:
            if unfold_on_failed:
                log.only_last_lines = None

            log.style = {'color': 'red'}

            raise subprocess.CalledProcessError(rc, p.args)

        if fold_on_success:
            log.only_last_lines = 0

    finally:
        log.root_parent.render()

    log.style = {'color': 'green'}

    return rc


def call_python(
    python_args: List[str],
    python_executable: Optional[str] = None,
    venv_dir: Optional[Path] = None,
    *args, **kwargs
) -> int:
    """
    가상환경을 활성화한 뒤 python 명령어를 실행합니다

    :param python_args: python 명령어에 전달할 인자들
    :param python_executable: 사용할 python 명령어 이름, None 이면 기본 python 을 사용
    :param venv_dir: 사용할 가상환경의 경로, None 이면 사용하지 않음
    :param args: call() 함수에 전달할 인자들
    :param kwargs: call() 함수에 전달할 인자들

    :return: 프로세스의 종료 코드
    """
    if not python_executable:
        python_executable = 'python'

    cmd = shlex.join([python_executable, '-u', *python_args])

    # venv 경로가 존재하면 적용하기
    if venv_dir and venv_dir.is_dir():
        cmd = f"(source {str(venv_dir.joinpath('bin', 'activate'))} && {cmd})"

        # venv 는 sh 에선 사용할 수 없기 때문에 bash 로 잡아줘야함
        if 'executable' not in kwargs:
            kwargs['executable'] = shutil.which('bash') or '/bin/bash'

    return call(
        cmd,
        shell=True,
        *args, **kwargs
    )
