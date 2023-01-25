import os
from typing import Callable
from shutil import rmtree
from tempfile import mkdtemp
from subprocess import call
from modules import sd_models

read_state_dict: Callable


# google drive is slow
def alternate_read_state_dict(checkpoint_file, *args, **kwargs):
    print('Copying model into temporary directory.')

    temp_dir = mkdtemp()
    copied_checkpoint_file = os.path.join(temp_dir, '')
    call(['cp', checkpoint_file, copied_checkpoint_file])

    print(f'Successfully copied model to {copied_checkpoint_file}')

    try:
        sd = read_state_dict(copied_checkpoint_file, *args, **kwargs)
        return sd
    finally:
        print('Discarding temporary model file.')
        rmtree(temp_dir, True)

if not sd_models.read_state_dict == alternate_read_state_dict:
    print('Applying alternate read_state_dict.')
    read_state_dict = sd_models.read_state_dict
    sd_models.read_state_dict = alternate_read_state_dict