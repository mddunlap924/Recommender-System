from pathlib import Path
import logging


logger = logging.getLogger(__name__)


def setup_logger(save_path: str) -> logging:
    # Update logger based on info in config. file
    filehandler = logging.FileHandler(Path(save_path), 'a')
    formatter = logging.Formatter('%(asctime)s::%(levelname)s::%(filename)s::'\
        '%(funcName)s::%(lineno)d::%(message)s')
    filehandler.setFormatter(formatter)
    logger = logging.getLogger()    # root logger - good to get it only once
    for hdlr in logger.handlers[:]:  # remove the existing file handlers
        if isinstance(hdlr, logging.FileHandler):
            logger.removeHandler(hdlr)
    logger.addHandler(filehandler)  # set the new handler
    # set the log level to INFO, DEBUG, as the default is ERROR
    logger.setLevel(logging.INFO)
    return logger


import collections
def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def log_cfg(cfg):
    logger.info('\n Log CFG YAML\n')
    cfg_flat = flatten(cfg)
    for key, value in cfg_flat.items():
        logger.info(f'{key}: {value}')
    logger.info('\n End of CFG YAML\n')
    return
