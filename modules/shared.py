from os import environ
from pathlib import Path

import __main__ as main

APP_DIR = Path(__file__).parent.parent

GDRIVE_MOUNT_DIR = APP_DIR.joinpath('drive')
GDRIVE_DIR = GDRIVE_MOUNT_DIR.joinpath('MyDrive')

IN_INTERACTIVE = not hasattr(main, '__file__')
IN_COLAB = 'IN_COLAB' in environ

try:
    from IPython import get_ipython
    IN_COLAB = IN_COLAB or 'google.colab' in str(get_ipython())
except ImportError:
    pass
