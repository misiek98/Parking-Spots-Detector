import json


def load_config(path: str) -> dict:
    """

    """
    with open(path, encoding='utf8') as cfg:
        config = json.load(cfg)
    return config
