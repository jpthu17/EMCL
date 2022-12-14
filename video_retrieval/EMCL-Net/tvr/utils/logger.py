import logging
import os
import sys


def setup_logger(name, save_dir, dist_rank, filename="log.txt"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.ERROR)
    # don't log results for the non-master process
    if dist_rank > 0:
        return logger
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    # formatter = logging.Formatter(f"[{dist_rank}]"+"[%(asctime)s %(name)s %(lineno)s %(levelname)s]: %(message)s")
    formatter = logging.Formatter("[%(asctime)s %(name)s %(lineno)s %(levelname)s]: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.propagate = False

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
