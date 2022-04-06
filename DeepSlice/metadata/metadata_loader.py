import json
from pathlib import Path
import os


def load_config() -> dict:
    """
    Loads the config file
    :return: the config file
    :rtype: dict
    """
    path = str(Path(__file__).parent) + os.sep
    with open(path + "config.json", "r") as f:
        config = json.loads(f.read())
    return config, path
