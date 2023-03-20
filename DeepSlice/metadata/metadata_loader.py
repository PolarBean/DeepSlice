import json
from pathlib import Path
import requests
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


def download_file(url: str, path: str):
    """
    Downloads a file from a url to a path

    :param url: the url of the file to download
    :type url: str
    :param path: the path to save the file to
    :type path: str
    """
    print("Downloading file from " + url + " to " + path)
    r = requests.get(url, allow_redirects=True)
    open(path, "wb").write(r.content)

def get_data_path(url_path_dict, path):
    """
    If the data is not present, download it from the DeepSlice github. Else return the path to the data.
    
    :param url_path_dict: a dictionary of a url and path to the data
    :type url_path_dict: dict
    :param path: the path to the DeepSlice metadata directory
    :type path: str
    :return: the path to the data
    :rtype: str
    """
    if not os.path.exists(path + url_path_dict["path"]):
        download_file(url_path_dict["url"], path + url_path_dict["path"])
    return path + url_path_dict["path"]
    
