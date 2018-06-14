from collections import defaultdict
import configparser
import json
from pathlib import Path


def load(configfile):
    configfile = Path(configfile)
    with configfile.open('r') as f:
        data = json.load(f)
    return data

def save(data, configfile):
    configfile = Path(configfile)
    with configfile.open('w') as f:
        json.dump(data, f, sort_keys=True, indent=4)
