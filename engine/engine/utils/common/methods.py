import json
from datetime import datetime
from functools import wraps
from os.path import expandvars
from typing import Union

import numpy as np
import yaml


def get_current_timestamp(use_hour=True):
    if use_hour:
        return datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        return datetime.now().strftime("%Y%m%d")


def load_yaml(yaml_path):
    def process_dict(dict_to_process):
        for key, item in dict_to_process.items():
            if isinstance(item, dict):
                dict_to_process[key] = process_dict(item)
            elif isinstance(item, str):
                dict_to_process[key] = expandvars(item)
            elif isinstance(item, list):
                dict_to_process[key] = process_list(item)
        return dict_to_process

    def process_list(list_to_process):
        new_list = []
        for item in list_to_process:
            if isinstance(item, dict):
                new_list.append(process_dict(item))
            elif isinstance(item, str):
                new_list.append(expandvars(item))
            elif isinstance(item, list):
                new_list.append(process_list(item))
            else:
                new_list.append(item)
        return new_list

    with open(yaml_path) as yaml_file:
        yaml_content = yaml.safe_load(yaml_file)

    return process_dict(yaml_content)


def load_json(path):
    with open(path) as f:
        content = json.load(f)
    return content


def save_yaml(content, path):
    with open(path, "w") as file:
        yaml.dump(content, file, sort_keys=False)


def create_json_file(output_filename, out_content):
    """create output json files"""
    with open(output_filename, "w") as out_file:
        json.dump(out_content, out_file, indent=1)


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = datetime.now()
        result = f(*args, **kw)
        te = datetime.now()
        print(f"func:{f.__qualname__} took: {te - ts}.")
        return result

    return wrap
