"""Module for basic configuration of the package logger."""

import logging


def configure_logger(name: str = "PyCBD", level: int | str = logging.WARNING):
    """Configure the logger for the PyCBD package

    :param name: The namespace of the loggers that need to be configured. By default, all PyCBD loggers will be
       configured, but you could also specify submodules, e.g., PyCBD.checkerboard_detection so only those loggers will
       be configured.
    :param level: The level that specifies which messages you will see. If you want to get additional information to
       be printed out you can set the level to logging.INFO.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
