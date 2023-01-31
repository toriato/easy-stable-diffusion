from pathlib import Path
from datetime import datetime
from IPython import get_ipython

APP_DIR = Path(__file__).parent.parent
GDRIVE_MOUNT_DIR = APP_DIR.joinpath('drive')
GDRIVE_DIR = GDRIVE_MOUNT_DIR.joinpath('MyDrive')

IN_COLAB = 'google.colab' in str(get_ipython())

LOG_PATH = APP_DIR.joinpath(
    'logs',
    datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log')
)
