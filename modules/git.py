import importlib.util
from . import shared, subprocess

if not importlib.util.find_spec('git'):
    if shared.IN_COLAB:
        subprocess.call(['pip', 'install', 'GitPython'])
    else:
        raise ImportError('GitPython 모듈을 찾을 수 없습니다')

from git import *  # type: ignore
