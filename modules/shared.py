from datetime import datetime
from os import environ
from pathlib import Path

from IPython import get_ipython

APP_DIR = Path(__file__).parent.parent
GDRIVE_MOUNT_DIR = APP_DIR.joinpath('drive')
GDRIVE_DIR = GDRIVE_MOUNT_DIR.joinpath('MyDrive')

IN_COLAB = 'IN_COLAB' in environ or 'google.colab' in str(get_ipython())
