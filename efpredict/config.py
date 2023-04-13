# efpredict paths for data and config files
# Kian Kordtomeikel (13/02/2023)
# https://github.com/echonet/dynamic/tree/master/echonet/

"""Sets paths based on configuration files."""

import configparser
import os
import types

_FILENAME = None
_PARAM = {}
for filename in ["efpredict.cfg",
                 ".efpredict.cfg",
                 os.path.expanduser("~/efpredict.cfg"),
                 os.path.expanduser("~/.efpredict.cfg"),
                 ]:
    if os.path.isfile(filename):
        _FILENAME = filename
        config = configparser.ConfigParser()
        with open(filename, "r") as f:
            config.read_string("[config]\n" + f.read())
            _PARAM = config["config"]
        break

CONFIG = types.SimpleNamespace(
    FILENAME=_FILENAME,
    DATA_DIR=_PARAM.get("DATA_DIR", "../Datasets/EchoNet-Dynamic/"),
    UNLABELLED_DIR=_PARAM.get("UNLABELLED_DIR", "../Datasets/EchoNet-LVH/"),
    PEDIATRIC_DIR=_PARAM.get("PEDIATRIC_DIR", "../Datasets/EchoNet-Pediatric/"))

