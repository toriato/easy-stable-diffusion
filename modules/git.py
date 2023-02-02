import importlib.util

from . import shared, subprocess

if not importlib.util.find_spec('git'):
    # 코랩에선 가상 서버 위에 돌아가기 때문에 사용자 동의 없이 패키지를 설치해도 상관 없지만
    # 개인 컴퓨터 같은 환경에선 오작동을 일으킬 수 있기 때문에 오류를 반환함
    if shared.IN_COLAB:
        subprocess.call(['pip', 'install', 'GitPython'])
    else:
        raise ImportError('GitPython 모듈을 찾을 수 없습니다')

from git import *  # type: ignore
