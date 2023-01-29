import json
import shutil
import shlex
import subprocess

from typing import List, Optional
from pathlib import Path

from modules import shared
from modules.log import Log


def call(*args, **kwargs) -> int:
    """
    하위 프로세스를 실행하고 모든 출력을 print() 함수로 전달합니다

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

    print_func = Log(
        json.dumps(p.args),
        parent=Log.context,
        child_style={'color': 'gray'}
    ).print if Log.context else print

    while p.poll() is None:
        assert p.stdout
        line = p.stdout.readline()
        if not line:
            continue

        print_func(line)

    rc = p.poll()
    assert rc is not None

    if rc != 0:
        raise subprocess.CalledProcessError(rc, p.args)

    return rc


def call_python(
    python_args: List[str],
    python_executable: Optional[str] = shared.PYTHON_EXECUTABLE,
    venv_dir: Optional[Path] = None,
    *args, **kwargs
) -> int:
    """
    가상환경을 활성화한 뒤 python 명령어를 실행합니다

    :param python_args: python 명령어에 전달할 인자들
    :param python_executable: python 명령어의 경로, None 이면 시스템에서 정의된 python 을 사용
    :param venv_dir: 가상환경의 경로, None 이면 사용하지 않음
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
