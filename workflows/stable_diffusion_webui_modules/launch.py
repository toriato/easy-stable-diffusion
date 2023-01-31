import shutil
from pathlib import Path

from modules import shared
from modules.subprocess import call, call_python

from .control import context, to_args


def setup_python() -> str:
    """
    사용자가 선택한 Python 이 존재하는지 확인하고 없으면 설치를 시도합니다

    :param context: 사용자 입력 컨텍스
    :return: Python 실행 바이너리 경로
    """

    python_executable = context['python_executable'].extract(context)
    assert isinstance(python_executable, str)

    # 코랩 환경이라면 APT 로 설치 시도하기
    if not shutil.which(python_executable) and shared.IN_COLAB:
        call(f'''
            apt install -y software-properties-common \\
            && add-apt-repository -y ppa:deadsnakes/ppa \\
            && apt update \\
            && apt install -y {python_executable} \\
            && curl --silent --show-error --retry 5 https://bootstrap.pypa.io/get-pip.py | {python_executable}
            ''')

    # 여전히 찾을 수 없다면 오류 반환하기
    if not shutil.which(python_executable):
        raise ValueError(f'Python 실행 바이너리를 찾을 수 없습니다')

    return python_executable


def setup_repository() -> git.Repo:
    repository = context['repository'].extract(context)
    assert isinstance(repository, str)

    path = Path('stable-diffusion-webui')

    if path.is_dir():
        repo = git.Repo(path)
    else:
        repo = git.Repo.clone_from(repository, path)

    # 레포지토리 커밋 체크아웃
    commit = context['repository_commit'].extract(context)
    assert isinstance(commit, str)
    if commit:
        repo.git.checkout(commit)

    return repo


def launch():
    try:
        repo = setup_repository()
        assert repo.working_dir

        python_executable = setup_python()

        args = to_args()

        call_python(
            ['-m', 'launch', *args],
            python_executable,
            cwd=repo.working_dir
        )
    except:
        raise
