import shlex
from pathlib import Path

from modules import shared
from modules.log import Log
from modules.utils import mount_google_drive

log = Log()

try:
    from IPython.display import display
    display(log.widget)
except ImportError:
    pass


def main(**kwargs):
    workspace = Path(kwargs['workspace'])

    if shared.IN_COLAB and kwargs['use_google_drive']:
        workspace = mount_google_drive() / workspace

    with log:
        from .stable_diffusion_webui_modules import environment as env
        env.options = env.Options(**{
            **kwargs,
            'workspace': workspace,
            'args': shlex.split(kwargs['args']),
            'extra_args': shlex.split(kwargs['extra_args'])
        })

        log.print(str(env.options))
