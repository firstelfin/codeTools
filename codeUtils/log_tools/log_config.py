#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   log_config.py
@Time    :   2024/08/18 19:20:54
@Author  :   firstElfin 
@Version :   1.0
@Desc    :   None
'''

import logging


class MyFormatter(logging.Formatter):
    # ANSI escape sequences for bold style
    COLORS = {
        'RESET': '\033[0m',
        'BOLD': '\033[1m',
        'GREEN': '\033[32m',
        'CYAN': '\033[36m',
        'RED': '\033[31m',
        'YELLOW': '\033[33m',
        'BLUE': '\033[34m',
    }

    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)

    def format(self, record):
        # Original formatted message
        original_message = super().format(record)

        # 设置时间打印为绿色
        timestamp = self.formatTime(record, self.datefmt) + f".{int(record.msecs):03d}"
        colored_timestamp = f"{self.COLORS['GREEN']}{timestamp}{self.COLORS['RESET']}"
        # Apply bold style to levelname
        bold_levelname = f"{self.COLORS['BOLD']}{record.levelname}{self.COLORS['RESET']}"
        # 设置name为黄色
        yellow_name = f"{self.COLORS['YELLOW']}{record.name}{self.COLORS['RESET']}"
        
        # Replace the levelname in the original message
        formatted_message = original_message.replace(record.levelname, bold_levelname)
        formatted_message = formatted_message.replace(timestamp, colored_timestamp)
        formatted_message = formatted_message.replace(record.name, yellow_name)

        return formatted_message


def setup_logger():

    config = {
        "version": 1,
        "disable_existing_loggers": False,

        "formatters": {
            "default": {
                "format": '%(asctime)s.%(msecs)03d | %(levelname)-7s  | %(name)s - %(message)s',
                "datefmt": '%Y-%m-%d %H:%M:%S'
            },
            "access": {
                "format": '%(asctime)s.%(msecs)03d | %(levelname)-7s  | %(name)s - %(message)s',
                "datefmt": '%Y-%m-%d %H:%M:%S'
            },
            "custom": {
                '()': MyFormatter,
                "format": '%(asctime)s.%(msecs)03d | %(levelname)-7s  | %(name)s - %(message)s',
                "datefmt": '%Y-%m-%d %H:%M:%S'
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "custom",
                "level": "INFO",
                "stream": "ext://sys.stdout"
            }
        },

        "loggers": {
            "uvicorn": {
                "handlers": ["console"],
                "level": "INFO",
                "propagate": False
            },
            "uvicorn.access": {
                "handlers": ["console"],
                "level": "INFO",
                "propagate": False
            }
        },
        "root": {
            "handlers": ["console"],
            "level": "INFO"
        }
    }

    # Apply logging configuration
    logging.config.dictConfig(config)
