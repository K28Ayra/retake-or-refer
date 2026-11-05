import yaml


def load_config(path="configs/default.yaml", realtime=None):
    with open(path) as f:
        config = yaml.safe_load(f)

    # the command line wins over the file
    if realtime is not None:
        config["realtime"] = realtime

    return config
