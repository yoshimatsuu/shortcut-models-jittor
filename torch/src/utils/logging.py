"""
Copyright (C) 2024 Yukara Ikemiya

Convenient modules for logging metrics.
"""

import sys
import typing as tp
import logging

import torch


class MetricsLogger:
    def __init__(self):
        self.counts = {}
        self.metrics = {}

    def add(self, metrics: tp.Dict[str, torch.Tensor]) -> None:
        for k, v in metrics.items():
            if k in self.counts.keys():
                self.counts[k] += 1
                self.metrics[k] += v.detach().clone()
            else:
                self.counts[k] = 1
                self.metrics[k] = v.detach().clone()

    def pop(self, mean: bool = True) -> tp.Dict[str, torch.Tensor]:
        metrics = {}
        for k, v in self.metrics.items():
            metrics[k] = v / self.counts[k] if mean else v

        # reset
        self.counts = {}
        self.metrics = {}

        return metrics


def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger()
    # log_format = '%(asctime)s | %(message)s'
    # formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    # file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    # stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger
