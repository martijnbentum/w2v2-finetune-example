from . import locations
import os
import shutil
from pathlib import Path

helper_files = ['preprocessor_config.json',
    'special_tokens_map.json',
    'tokenizer_config.json',
    'vocab.json']


def helper_files_present(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    for f in helper_files:
        if not (checkpoint_dir / f).exists():
            return False
    return True

def add_helper_files(checkpoint_dir, transcription = None,
    helper_files_directory = 'helper_files/'):
    helper_files_directory = Path(helper_files_directory)
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f'{checkpoint_dir} does not exist')
    if helper_files_present(checkpoint_dir):
        print(f'Helper files already present in {checkpoint_dir}')
        return
    for f in helper_files:
        copy_file(f, helper_files_directory, checkpoint_dir)
    print(f'Helper files added to {checkpoint_dir}')

def copy_file(filename, source, destination):
    source = Path(source) / filename
    destination = Path(destination) / filename
    print(f'Copying {source} to {destination}')
    shutil.copyfile(source, destination)
